# ranked_cocoeval.py
"""
Extended COCOeval class that supports multiple predictions per image with video-level ranking.
"""

import numpy as np
import time
import copy
from collections import defaultdict
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


class RankedCOCOeval(COCOeval):
    """
    Extended COCOeval that handles multiple predictions per image (via prediction_id)
    and provides video-level ranking capabilities.
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType="segm"):
        """Initialize with support for prediction_id tracking."""
        super().__init__(cocoGt, cocoDt, iouType)
        self.prediction_ids = []

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params.
        Modified to support prediction_id in detection keys.
        """

        def _toMask(anns, coco):
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)

        # MODIFIED: Include prediction_id in the key
        for dt in dts:
            prediction_id = dt.get(
                "prediction_id", 0
            )  # For oracle we do not store prediction id
            self._dts[(dt["image_id"], dt["category_id"], prediction_id)].append(dt)

        self.evalImgs = defaultdict(list)
        self.eval = {}

    def evaluate(self):
        """
        Run per image evaluation on given images and store results.
        Modified to handle prediction_id for multiple predictions per image.
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params

        if p.useSegm is not None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print(
                "useSegm (deprecated) is not None. Running {} evaluation".format(
                    p.iouType
                )
            )

        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))

        if p.useCats:
            p.catIds = list(np.unique(p.catIds))

        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()

        catIds = p.catIds if p.useCats else [-1]

        # MODIFIED: Track all prediction_ids
        self.prediction_ids = sorted(
            set(
                dt.get("prediction_id", 0)
                for dt in self.cocoDt.loadAnns(self.cocoDt.getAnnIds())
            )
        )

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif p.iouType == "keypoints":
            computeIoU = self.computeOks

        # MODIFIED: Compute IoU for each (image, category, prediction_id) tuple
        self.ious = {
            (imgId, catId, prediction_id): computeIoU(imgId, catId, prediction_id)
            for imgId in p.imgIds
            for catId in catIds
            for prediction_id in self.prediction_ids
        }

        maxDet = p.maxDets[-1]

        # MODIFIED: Evaluate for each (image, category, prediction_id) tuple
        self.evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet, prediction_id)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
            for prediction_id in self.prediction_ids
        ]

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId, prediction_id):
        """
        Compute IoU between detections and ground truths.
        Modified to accept prediction_id parameter.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId, prediction_id]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId, prediction_id]]

        if len(gt) == 0 and len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet, prediction_id):
        """
        Perform evaluation for single category and image.
        Modified to accept prediction_id and include it in results.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId, prediction_id]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId, prediction_id]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < aRng[0] or g["area"] > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]

        ious = (
            self.ious[imgId, catId, prediction_id][:, gtind]
            if len(self.ious[imgId, catId, prediction_id]) > 0
            else self.ious[imgId, catId, prediction_id]
        )

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        if ious[dind, gind] < iou:
                            continue
                        iou = ious[dind, gind]
                        m = gind
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]

        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape(
            (1, len(dt))
        )
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        # MODIFIED: Include prediction_id in results
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gtIg,
            "dtIgnore": dtIg,
            "prediction_id": prediction_id,
        }

    def accumulate_per_video(self, video_to_imgs):
        """
        Compute AP for each (video_id, prediction_id) pair for ranking.

        Args:
            video_to_imgs: dict mapping video_id -> list of image_ids

        Returns:
            video_rankings: dict {
                video_id: {
                    'scores': {pred_id: AP},
                    'worst_pid': int,
                    'median_pid': int,
                    'best_pid': int
                }
            }
        """
        print("Computing per-video rankings...")
        tic = time.time()

        if not self.evalImgs:
            print("Please run evaluate() first")
            return {}

        # Build reverse mapping: img_id -> video_id
        img_to_video = {}
        for video_id, img_ids in video_to_imgs.items():
            for img_id in img_ids:
                img_to_video[img_id] = video_id

        video_rankings = {}

        for video_id, img_ids in video_to_imgs.items():
            img_ids_set = set(img_ids)
            scores = {}

            # For each prediction_id, compute AP on this video
            for pred_id in self.prediction_ids:
                # Filter evalImgs to this video and prediction_id
                video_pred_evalImgs = [
                    e
                    for e in self.evalImgs
                    if e is not None
                    and e["image_id"] in img_ids_set
                    and e["prediction_id"] == pred_id
                ]

                if not video_pred_evalImgs:
                    scores[pred_id] = 0.0
                    continue

                # Compute AP for this subset
                ap = self._compute_ap_from_evalImgs(video_pred_evalImgs)
                scores[pred_id] = ap

            # Rank predictions
            if not scores:
                video_rankings[video_id] = {
                    "scores": {},
                    "worst_pid": 0,
                    "median_pid": 0,
                    "best_pid": 0,
                }
                continue

            sorted_items = sorted(scores.items(), key=lambda x: x[1])
            worst_pid = sorted_items[0][0]
            best_pid = sorted_items[-1][0]
            median_pid = sorted_items[len(sorted_items) // 2][0]

            video_rankings[video_id] = {
                "scores": scores,
                "worst_pid": worst_pid,
                "median_pid": median_pid,
                "best_pid": best_pid,
            }

        toc = time.time()
        print(f"Per-video ranking DONE (t={toc - tic:.2f}s).")
        return video_rankings

    def _compute_ap_from_evalImgs(self, evalImgs_subset):
        """
        Compute AP from a subset of evalImgs (helper for accumulate_per_video).
        Simplified version of accumulate() for a single metric.
        """
        p = self.params

        # Gather all detections from this subset
        dtScores = []
        dtMatches_list = []
        dtIgnore_list = []
        gtIgnore_list = []

        for e in evalImgs_subset:
            if e is None:
                continue
            maxDet = p.maxDets[-1]
            dtScores.extend(e["dtScores"][:maxDet])
            dtMatches_list.append(e["dtMatches"][:, :maxDet])
            dtIgnore_list.append(e["dtIgnore"][:, :maxDet])
            gtIgnore_list.append(e["gtIgnore"])

        if not dtScores:
            return 0.0

        # Sort by score
        dtScores = np.array(dtScores)
        inds = np.argsort(-dtScores, kind="mergesort")
        # dtScoresSorted = dtScores[inds]

        # Concatenate and reorder
        dtm = np.concatenate(dtMatches_list, axis=1)[:, inds]
        dtIg = np.concatenate(dtIgnore_list, axis=1)[:, inds]
        gtIg = np.concatenate(gtIgnore_list)

        # Count non-ignored GTs
        npig = np.count_nonzero(gtIg == 0)
        if npig == 0:
            return 0.0

        # Compute TP/FP
        tps = np.logical_and(dtm, np.logical_not(dtIg))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

        # Cumulative sums
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

        # Compute precision at all IoU thresholds and average
        aps = []
        T = len(p.iouThrs)
        R = len(p.recThrs)

        for t in range(T):
            tp = tp_sum[t]
            fp = fp_sum[t]

            if len(tp) == 0:
                aps.append(0.0)
                continue

            # Precision-recall
            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))

            # Smooth precision (monotonic decreasing)
            pr = pr.tolist()
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            # Interpolate at standard recall levels
            q = np.zeros(R)
            inds = np.searchsorted(rc, p.recThrs, side="left")
            for ri, pi in enumerate(inds):
                if pi < len(pr):
                    q[ri] = pr[pi]

            # AP for this IoU threshold
            aps.append(np.mean(q))

        # Return mean AP across IoU thresholds
        return np.mean(aps)

    def accumulate_ranked(self, video_rankings, video_to_imgs, ranking_type="best"):
        """
        Accumulate using ranked predictions (worst/median/best per video).

        Args:
            video_rankings: Output from accumulate_per_video()
            video_to_imgs: dict mapping video_id -> list of image_ids
            ranking_type: 'worst', 'median', or 'best'
        """
        print(f"Accumulating {ranking_type} predictions...")
        tic = time.time()

        if not self.evalImgs:
            print("Please run evaluate() first")
            return

        # Build reverse mapping
        img_to_video = {}
        for video_id, img_ids in video_to_imgs.items():
            for img_id in img_ids:
                img_to_video[img_id] = video_id

        # Determine which prediction_id to use for each video
        video_to_selected_pid = {}
        for video_id, ranking in video_rankings.items():
            if ranking_type == "worst":
                selected_pid = ranking["worst_pid"]
            elif ranking_type == "median":
                selected_pid = ranking["median_pid"]
            else:  # best
                selected_pid = ranking["best_pid"]
            video_to_selected_pid[video_id] = selected_pid

        # Rebuild evalImgs with correct (K, A, I) structure
        p = self.params
        catIds = p.catIds if p.useCats else [-1]

        evaluated_videos = set(video_rankings.keys())
        evaluated_imgIds = [
            imgId for imgId in p.imgIds if img_to_video.get(imgId) in evaluated_videos
        ]

        # Dimensions
        K = len(catIds)
        A = len(p.areaRng)
        I = len(p.imgIds)
        P = len(self.prediction_ids)

        # Create new filtered list maintaining (K, A, I) indexing
        filtered_evalImgs = []

        for k_idx, catId in enumerate(catIds):
            for a_idx, areaRng in enumerate(p.areaRng):
                for i_idx, imgId in enumerate(p.imgIds):
                    # Determine selected prediction_id for this image
                    video_id = img_to_video.get(imgId)

                    if video_id is None or video_id not in video_to_selected_pid:
                        # Image not in any video or no selected prediction
                        filtered_evalImgs.append(None)
                        continue

                    selected_pid = video_to_selected_pid[video_id]

                    # Find corresponding element in original evalImgs
                    # Original index: k*A*I*P + a*I*P + i*P + p
                    p_idx = self.prediction_ids.index(selected_pid)
                    orig_idx = k_idx * A * I * P + a_idx * I * P + i_idx * P + p_idx

                    if orig_idx < len(self.evalImgs):
                        filtered_evalImgs.append(self.evalImgs[orig_idx])
                    else:
                        filtered_evalImgs.append(None)

        # Verify length
        expected_len = K * A * I
        if len(filtered_evalImgs) != expected_len:
            print(
                f"WARNING: Filtered list length {len(filtered_evalImgs)} != expected {expected_len}"
            )

        # Temporarily replace evalImgs
        original_evalImgs = self.evalImgs
        self.evalImgs = filtered_evalImgs

        # Run standard accumulate
        super().accumulate()

        # Restore original
        self.evalImgs = original_evalImgs

        toc = time.time()
        print(f"DONE (t={toc - tic:.2f}s).")
