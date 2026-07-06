import argparse
import datetime
import json
import os
import pathlib
import shutil

import pandas as pd
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from einops import rearrange
from tqdm import tqdm

from waymo_open_dataset import dataset_pb2 as open_dataset

tf.compat.v1.enable_eager_execution()


class WaymoCOCOConverter:
    def __init__(
        self,
        image_dir=None,
        video_dir=None,
        df_metadata_path=None,
        image_prefix=None,
        write_image=False,
        add_waymo_info=False,
        add_coco_info=True,
        img_size=512,
    ):
        """
        Parameters
        ----------
        add_waymo_info : bool
            include additional information out of original COCO format.
        add_coco_info : bool
            include information in original COCO format,
            but out of Waymo Open Dataset.
            if set to False, COCO compatibility breaks.
        """

        self.image_dir = image_dir
        self.video_dir = video_dir
        self.df_metadata_path = df_metadata_path

        self.image_prefix = image_prefix
        self.write_image = write_image
        self.add_waymo_info = add_waymo_info
        self.add_coco_info = add_coco_info

        self.init_waymo_dataset_proto_info()
        self.init_coco_format_info()

        self.img_index = 0
        self.annotation_index = 0
        self.clip_index = 0
        self.img_dicts = []
        self.annotation_dicts = []
        self.df_metadata_rows = []
        self.img_size = img_size

        # These 2 are populated only at the end
        self.coco_output_dict = None
        self.df_metadata = None

    def init_waymo_dataset_proto_info(self):
        # these lists should correspond to label.proto and dataset.proto in
        # https://github.com/waymo-research/waymo-open-dataset/
        self.waymo_class_mapping = [
            "TYPE_UNKNOWN",
            "TYPE_VEHICLE",
            "TYPE_PEDESTRIAN",
            "TYPE_SIGN",
            "TYPE_CYCLIST",
        ]
        self.waymo_camera_names = {
            0: "UNKNOWN",
            1: "FRONT",
            2: "FRONT_LEFT",
            3: "FRONT_RIGHT",
            4: "SIDE_LEFT",
            5: "SIDE_RIGHT",
        }

    def init_coco_format_info(self):
        self.dataset_info = {
            "year": 2020,
            "version": "v1.2_20200409",
            "description": "Waymo Open Dataset 2D Detection",
            "contributor": "Waymo LLC",
            "url": "https://waymo.com/open/",
            "date_created": datetime.datetime.utcnow().isoformat(" "),
        }
        self.licenses = [
            {
                "id": 1,
                "name": "Waymo Dataset License Agreement for Non-Commercial Use",
                "url": "https://waymo.com/open/terms/",
            }
        ]
        # Waymo Open Dataset categories for "ALL_NS" setting
        # (all Object Types except signs)
        # Note that ids are different from those of waymo_class_mapping
        self.target_categories = [
            {
                "id": 1,
                "name": "TYPE_VEHICLE",
                "supercategory": "vehicle",
            },
            {
                "id": 2,
                "name": "TYPE_PEDESTRIAN",
                "supercategory": "person",
            },
            {
                "id": 3,
                "name": "TYPE_CYCLIST",
                "supercategory": "bike_plus",
            },
            # {
            #     "id" : ,
            #     "name" : "TYPE_SIGN",
            #     "supercategory" : "outdoor",
            # },
            # {
            #     "id" : ,
            #     "name" : "TYPE_UNKNOWN",
            #     "supercategory" : "unknown",
            # },
        ]

    def process_sequences(self, tfrecord_paths):
        for parent_video_id, tfrecord_path in enumerate(
            tqdm(sorted(tfrecord_paths), desc="Sequences")
        ):
            sequence_data = tf.data.TFRecordDataset(
                filenames=str(tfrecord_path), compression_type=""
            )
            sequence_frames = list(sequence_data)
            number_of_frames_in_sequence = len(sequence_frames)
            number_total_frames = 16
            number_subvideos = number_of_frames_in_sequence // number_total_frames

            for subvideo_idx in range(number_subvideos):
                idx_start = subvideo_idx * number_total_frames
                idx_end = idx_start + number_total_frames

                # Collect frames organized by camera
                camera_frames = {}

                for local_frame_idx, global_frame_idx in enumerate(
                    range(idx_start, idx_end)
                ):
                    frame = open_dataset.Frame()
                    frame.ParseFromString(
                        bytearray(sequence_frames[global_frame_idx].numpy())
                    )

                    # Process each camera image
                    for camera_image in frame.images:
                        camera_name = camera_image.name

                        # Initialize camera dict with unique clip_id
                        if camera_name not in camera_frames:
                            camera_frames[camera_name] = {
                                "clip_id": self.clip_index,
                                "frames": [],
                                "image_ids": [],
                                "original_height": [],
                                "original_width": [],
                            }
                            self.clip_index += 1  # ← Increment per camera view

                        clip_id = camera_frames[camera_name]["clip_id"]

                        frame_tensor, img_id, img_height, img_width = self.process_img(
                            clip_id=clip_id,
                            camera_image=camera_image,
                            frame=frame,
                            global_frame_idx=global_frame_idx,
                            parent_video_id=parent_video_id,
                            subvideo_idx=subvideo_idx,
                            local_frame_idx=local_frame_idx,
                        )

                        camera_frames[camera_name]["frames"].append(frame_tensor)
                        camera_frames[camera_name]["image_ids"].append(img_id)
                        camera_frames[camera_name]["original_height"].append(img_height)
                        camera_frames[camera_name]["original_width"].append(img_width)

                        # Add annotations with correct clip_id
                        for camera_label in frame.camera_labels:
                            if camera_label.name != camera_image.name:
                                continue
                            self.add_coco_annotation_dict(
                                camera_label=camera_label,
                                parent_video_id=parent_video_id,
                                clip_id=clip_id,
                                subvideo_idx=subvideo_idx,
                                local_frame_idx=local_frame_idx,
                                global_frame_idx=global_frame_idx,
                            )
                        self.img_index += 1

                # Save videos (clip_id already assigned)
                self.save_videos(
                    camera_frames, parent_video_id, subvideo_idx, idx_start, idx_end
                )

    def process_img(
        self,
        clip_id,
        camera_image,
        frame,
        global_frame_idx,
        parent_video_id,
        subvideo_idx,
        local_frame_idx,
    ):

        img_id = self.img_index
        img_filename = f"{img_id}.jpg"

        img = tf.image.decode_jpeg(contents=camera_image.image).numpy().copy()
        img_height = img.shape[0]
        img_width = img.shape[1]

        if self.write_image:
            if self.image_dir is None:
                raise ValueError("image_dir is None. Cannot save images.")
            img_path = os.path.join(self.image_dir, img_filename)
            with open(file=img_path, mode="wb") as f:
                f.write(bytearray(camera_image.image))

        frame_tensor = torch.tensor(data=img)  # Appended for videos

        # In COCO, 'file_name' is usually relative; we store just the filename.
        self.add_coco_img_dict(
            file_name=img_filename,
            clip_id=clip_id,
            height=img_height,
            width=img_width,
            parent_video_id=parent_video_id,
            global_frame_idx=global_frame_idx,
            camera_id=int(camera_image.name),
            frame=frame,
            subvideo_idx=subvideo_idx,
            local_frame_idx=local_frame_idx,
        )

        return frame_tensor, img_id, img_height, img_width

    def add_coco_img_dict(
        self,
        file_name,
        clip_id=None,
        height=None,
        width=None,
        parent_video_id=None,
        global_frame_idx=None,
        camera_id=None,
        frame=None,
        subvideo_idx=None,
        local_frame_idx=None,
    ):

        if height is None or width is None:
            raise ValueError

        img_dict = {
            "id": self.img_index,
            "clip_id": clip_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 1,
        }

        if self.add_coco_info:
            img_dict["flickr_url"] = ""
            img_dict["coco_url"] = ""
            img_dict["date_captured"] = ""

        if self.add_waymo_info:
            img_dict["context_name"] = frame.context.name
            img_dict["timestamp_micros"] = frame.timestamp_micros
            img_dict["camera_id"] = camera_id
            img_dict["time_of_day"] = frame.context.stats.time_of_day
            img_dict["location"] = frame.context.stats.location
            img_dict["weather"] = frame.context.stats.weather

            img_dict["parent_video_id"] = parent_video_id
            img_dict["subvideo_idx"] = subvideo_idx

            img_dict["local_frame_idx"] = local_frame_idx
            img_dict["global_frame_idx"] = global_frame_idx

        self.img_dicts.append(img_dict)

    def add_coco_annotation_dict(
        self,
        camera_label,
        parent_video_id=None,
        clip_id=None,
        subvideo_idx=None,
        local_frame_idx=None,
        global_frame_idx=None,
    ):
        annotation_dicts = []
        for box_label in camera_label.labels:
            category_name_to_id = {
                category["name"]: category["id"] for category in self.target_categories
            }
            category_name = self.waymo_class_mapping[box_label.type]
            category_id = category_name_to_id[category_name]
            if category_id not in [1, 2, 3]:
                print(f"category id {category_id} not in target categories, skipping.")
            width = box_label.box.length  # box.length: dim x
            height = box_label.box.width  # box.width: dim y
            x1 = box_label.box.center_x - width / 2
            y1 = box_label.box.center_y - height / 2

            annotation_dict = {
                "id": self.annotation_index,
                "image_id": self.img_index,
                "category_id": category_id,
                "segmentation": None,
                "area": width * height,
                "bbox": [x1, y1, width, height],
                "iscrowd": 0,
            }
            if self.add_waymo_info:
                annotation_dict["track_id"] = box_label.id
                annotation_dict["det_difficult"] = box_label.detection_difficulty_level
                annotation_dict["track_difficult"] = box_label.tracking_difficulty_level
                annotation_dict["parent_video_id"] = parent_video_id
                annotation_dict["clip_id"] = clip_id
                annotation_dict["subvideo_idx"] = subvideo_idx
                annotation_dict["local_frame_idx"] = local_frame_idx
                annotation_dict["global_frame_idx"] = global_frame_idx
            annotation_dicts.append(annotation_dict)
            self.annotation_index += 1
        self.annotation_dicts.extend(annotation_dicts)

    def write_coco_annotations_json(self, label_path, json_indent=None):
        if self.coco_output_dict is None:
            raise ValueError(
                "COCO annotations not assembled. Call assemble_coco_annotations() first."
            )

        with open(file=label_path, mode="w") as f:
            # dump with a trick for rounding float
            json.dump(
                obj=json.loads(
                    s=json.dumps(obj=self.coco_output_dict),
                    parse_float=lambda x: round(float(x), 6),
                ),
                fp=f,
                indent=json_indent,
                sort_keys=False,
            )

    def save_videos(
        self, camera_frames, parent_video_id, subvideo_idx, idx_start, idx_end
    ):
        """Save videos for each camera in the subvideo and log metadata."""
        if self.video_dir is None:
            raise ValueError("video_dir is None. Cannot save videos.")

        for camera_name, camera_data in camera_frames.items():
            clip_id = camera_data["clip_id"]
            frames = camera_data["frames"]
            image_ids = camera_data["image_ids"]
            original_height = camera_data["original_height"]
            original_width = camera_data["original_width"]
            # Stack to (T, H, W, C)
            video_tensor = torch.stack(tensors=frames, dim=0)
            # print(f"shape video_tensor before resizing: {video_tensor.shape}")

            video_tensor = self.resize_video(
                video_tensor=video_tensor,
                img_size=self.img_size,
                method="interpolate",
            )
            # print(f"shape video_tensor after resizing: {video_tensor.shape}")

            # Use clip_id as the video filename.
            video_filename = f"{clip_id}.pt"
            video_path = os.path.join(self.video_dir, video_filename)

            # Save the video tensor
            if self.write_image:
                torch.save(obj=video_tensor, f=video_path)

            # Check that every item in originial height is the same
            if not len(set(original_height)) == 1:
                raise ValueError("Original heights are not the same within the video.")
            if not len(set(original_width)) == 1:
                raise ValueError("Original widths are not the same within the video.")

            # Append metadata row for this (sequence, subvideo, camera)
            global_frame_indices = list(range(idx_start, idx_end))

            row = {
                "parent_video_id": parent_video_id,
                "clip_id": clip_id,
                "subvideo_idx": subvideo_idx,
                "camera_name": int(camera_name),
                "video_filename": video_filename,  # relative, like COCO file_name
                "num_frames": int(video_tensor.shape[0]),
                "image_ids": image_ids,
                "global_frame_indices": global_frame_indices,
                "original_height": original_height,
                "original_width": original_width,
            }
            self.df_metadata_rows.append(row)

    def resize_video(self, video_tensor, img_size, method):
        video_tensor = rearrange(tensor=video_tensor, pattern="T H W C -> T C H W")
        if method == "resize":
            resized = tvF.resize(img=video_tensor, size=[img_size, img_size])
        elif method == "interpolate":
            resized = F.interpolate(
                input=video_tensor,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise ValueError(
                f"Unsupported resize method '{method}'. Expected 'resize' or 'interpolate'."
            )
        return resized

    def assemble_coco_annotations(self):
        """Assemble COCO annotations dictionary and store in self."""
        self.coco_output_dict = {
            "info": self.dataset_info,
            "licenses": self.licenses,
            "categories": self.target_categories,
            "images": self.img_dicts,
        }
        if self.annotation_dicts:
            self.coco_output_dict["annotations"] = self.annotation_dicts
        if self.add_waymo_info:
            self.coco_output_dict["camera_names"] = self.waymo_camera_names

    def assemble_df_metadata(self):
        """Assemble metadata DataFrame and store in self."""
        self.df_metadata = pd.DataFrame(data=self.df_metadata_rows)

    def check_annotations_and_df_validity(self):
        """Check validity of both COCO annotations and metadata."""
        if self.coco_output_dict is None:
            raise ValueError(
                "COCO annotations not assembled. Call assemble_coco_annotations() first."
            )
        if self.df_metadata is None:
            raise ValueError(
                "Metadata DataFrame not assembled. Call assemble_df_metadata() first."
            )

        # Check COCO annotations validity
        images = self.coco_output_dict["images"]
        annotations = self.coco_output_dict["annotations"]

        n_images = len(images)
        n_img_ids_from_images = len(set([img["id"] for img in images]))
        n_img_ids_from_annotations = len(set([ann["image_id"] for ann in annotations]))
        n_rows_times_16 = len(self.df_metadata) * 16

        all_image_ids_from_df = []
        for _, row in self.df_metadata.iterrows():
            all_image_ids_from_df.extend(row["image_ids"])
        n_unique_img_ids_from_df = len(set(all_image_ids_from_df))

        # print these values
        print("###### CHECK ON IMAGE IDS ######")
        print(f"len(self.coco_output_dict['images']): {n_images}")
        print(f"n img ids from images: {n_img_ids_from_images}")
        print(f"self img index: {self.img_index}")
        print(f"n rows in dataframe * 16: {n_rows_times_16}")
        print(f"n unique image IDs from dataframe: {n_unique_img_ids_from_df}")
        print(f"n img ids from annotations: {n_img_ids_from_annotations}")

        # These 3 values should be equal
        assert (
            n_images
            == n_img_ids_from_images
            == self.img_index
            == n_rows_times_16
            == n_unique_img_ids_from_df
        ), "Mismatch in number of images and image IDs."

        # Check the number of img ids from annotations is less
        assert n_img_ids_from_annotations <= n_images, (
            "Number of image IDs from annotations exceeds number of images."
        )

        # Print proportion of images which have annotations
        print(
            f"Proportion of images with annotations: {n_img_ids_from_annotations}/{n_images} = {n_img_ids_from_annotations / n_images:.2f}"
        )

        ######################### CHECK ON CLIP IDS #########################
        print()
        print("###### CHECK ON CLIP IDS ######")
        print(
            f"Number of unique clip_ids in df_metadata: {self.df_metadata['clip_id'].nunique()}"
        )
        print(f"Len df_metadata_rows: {len(self.df_metadata_rows)}")
        print(f"Self clip_index: {self.clip_index}")
        # assert these 3 values match
        assert (
            self.df_metadata["clip_id"].nunique()
            == len(self.df_metadata_rows)
            == self.clip_index
        ), "Mismatch in number of clip_ids."

        # Check matching clip ids between annotations and images
        print()
        print(
            "###### CHECK ON CLIP ID CONSISTENCY BETWEEN ANNOTATIONS AND IMAGES ######"
        )
        image_id_to_clip_id = {img["id"]: img["clip_id"] for img in images}

        mismatches = []
        for ann in annotations:
            ann_image_id = ann["image_id"]  # get the img id for this annotation
            ann_clip_id = ann["clip_id"]  # get the clip id for this annotation

            img_clip_id = image_id_to_clip_id[
                ann_image_id
            ]  # get the clip id from the corresponding img
            if ann_clip_id != img_clip_id:
                mismatches.append(
                    f"Annotation {ann['id']}: clip_id mismatch (ann={ann_clip_id}, img={img_clip_id})"
                )

        if mismatches:
            print(f"Found {len(mismatches)} clip_id mismatches:")
            for mismatch in mismatches[:10]:  # print first 10
                print(f"  {mismatch}")
            raise ValueError("clip_id mismatches found between annotations and images")

        ######################### CHECK ON PARENT VIDEO ID AND CLIP ID CONSISTENCY #########################
        print()
        print(
            "###### CHECK ON PARENT VIDEO ID AND CLIP ID CONSISTENCY ACROSS DF, IMAGES, AND ANNOTATIONS ######"
        )

        # Create mappings from image_id
        image_id_to_parent_and_clip = {
            img["id"]: (img["parent_video_id"], img["clip_id"]) for img in images
        }

        mismatches = []
        for _, row in self.df_metadata.iterrows():
            expected_parent_id = row["parent_video_id"]
            expected_clip_id = row["clip_id"]

            # Check all 16 image_ids for this row
            for img_id in row["image_ids"]:
                actual_parent_id, actual_clip_id = image_id_to_parent_and_clip[img_id]

                if (
                    actual_parent_id != expected_parent_id
                    or actual_clip_id != expected_clip_id
                ):
                    mismatches.append(
                        f"Image {img_id}: expected (parent={expected_parent_id}, clip={expected_clip_id}), "
                        f"got (parent={actual_parent_id}, clip={actual_clip_id})"
                    )

        if mismatches:
            print(f"Found {len(mismatches)} mismatches in images:")
            for mismatch in mismatches[:10]:
                print(f"  {mismatch}")
            raise ValueError("parent_video_id/clip_id mismatches found in images")

        # Check annotations consistency
        ann_mismatches = []
        for ann in annotations:
            ann_image_id = ann["image_id"]
            ann_parent_id = ann["parent_video_id"]
            ann_clip_id = ann["clip_id"]

            img_parent_id, img_clip_id = image_id_to_parent_and_clip[ann_image_id]

            if ann_parent_id != img_parent_id or ann_clip_id != img_clip_id:
                ann_mismatches.append(
                    f"Annotation {ann['id']}: expected (parent={img_parent_id}, clip={img_clip_id}), "
                    f"got (parent={ann_parent_id}, clip={ann_clip_id})"
                )

        if ann_mismatches:
            print(f"Found {len(ann_mismatches)} mismatches in annotations:")
            for mismatch in ann_mismatches[:10]:
                print(f"  {mismatch}")
            raise ValueError("parent_video_id/clip_id mismatches found in annotations")

        print("All parent_video_id and clip_id are consistent!")

    def write_df_metadata(self):
        """Write df_metadata to CSV at self.df_metadata_path."""
        if self.df_metadata_path is None:
            raise ValueError("df_metadata_path is None. Cannot write metadata.")
        if self.df_metadata is None:
            raise ValueError(
                "Metadata DataFrame not assembled. Call assemble_df_metadata() first."
            )

        self.df_metadata.to_csv(path_or_buf=self.df_metadata_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfrecord_dir", required=True, type=str
    )  # source Waymo TFRecords
    parser.add_argument("--work_dir", default=".", type=str)
    parser.add_argument(
        "--image_dirname", required=True, type=str
    )  # output images directory name
    parser.add_argument(
        "--video_dirname", required=True, type=str
    )  # output video tensors directory name
    parser.add_argument("--image_filename_prefix", default=None, type=str)
    parser.add_argument("--label_dirname", default="annotations", type=str)
    parser.add_argument("--label_filename", required=True, type=str)
    parser.add_argument("--json_indent", default=None, type=int)
    parser.add_argument("--add_waymo_info", action="store_true")
    parser.add_argument("--write_image", action="store_true")
    parser.add_argument(
        "--sequence_limit",
        default=None,
        type=int,
        help="limit number of sequences. useful for debug.",
    )
    args = parser.parse_args()

    # Create directories to save images/videos/annotations
    image_dir = os.path.join(args.work_dir, args.image_dirname)  # export images here
    label_dir = os.path.join(
        args.work_dir, args.label_dirname
    )  # export annotations here
    video_dir = os.path.join(args.work_dir, args.video_dirname)  # export videos here

    print(f"image_dir: {image_dir}")
    print(f"label_dir: {label_dir}")
    print(f"video_dir: {video_dir}")

    # Create paths for annotation.json and df_metadata.csv
    label_path = os.path.join(label_dir, args.label_filename)
    df_metadata_path = os.path.join(video_dir, "df_metadata.csv")
    print(f"label_path: {label_path}")
    print(f"df_metadata_path: {df_metadata_path}")

    # if these folders exist remove them with shutil
    if os.path.exists(image_dir):
        shutil.rmtree(path=image_dir)
        print(f"Removed existing image_dir: {image_dir}")
    if os.path.exists(video_dir):
        shutil.rmtree(path=video_dir)
        print(f"Removed existing video_dir: {video_dir}")
    # Only remove the annotations file if it exist (json)
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Removed existing label_path: {label_path}")

    def ensure_dir(path, label):
        if not os.path.isdir(path):
            os.makedirs(name=path, exist_ok=True)
            print(f"Created {label}: {path}")
        else:
            os.makedirs(name=path, exist_ok=True)
            print(f"{label} already exists: {path}")

    ensure_dir(path=image_dir, label="image_dir")
    ensure_dir(path=label_dir, label="label_dir")
    ensure_dir(path=video_dir, label="video_dir")

    tfrecord_list = list(
        sorted(pathlib.Path(args.tfrecord_dir).glob(pattern="*.tfrecord"))
    )
    print(f"Found {len(tfrecord_list)} tfrecord files.")

    # if args.sequence_limit is not None:
    #     tfrecord_list = tfrecord_list[:args.sequence_limit]
    # tfrecord_list = tfrecord_list[:5]

    waymo_converter = WaymoCOCOConverter(
        image_dir=image_dir,
        video_dir=video_dir,
        df_metadata_path=df_metadata_path,
        image_prefix=args.image_filename_prefix,
        write_image=args.write_image,
        add_waymo_info=args.add_waymo_info,
        img_size=512,
    )

    waymo_converter.process_sequences(
        tfrecord_paths=tfrecord_list
    )  # Process videos, images, annotations

    waymo_converter.assemble_coco_annotations()

    waymo_converter.assemble_df_metadata()

    waymo_converter.check_annotations_and_df_validity()

    # Write COCO annotation JSON
    waymo_converter.write_coco_annotations_json(
        label_path=label_path,
        json_indent=args.json_indent,
    )
    # Write df_metadata CSV
    waymo_converter.write_df_metadata()

    # Print message for completion
    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()
