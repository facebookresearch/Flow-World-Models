import os
from collections import defaultdict
import imageio.v2 as imageio
import torch
from einops import rearrange
from torchvision.io import write_png
from sklearn.decomposition import PCA
import logging


def write_video(path, video_tensor, fps):
    video_tensor = video_tensor.detach().cpu()
    if video_tensor.dtype != torch.uint8:
        video_tensor = video_tensor.clamp(0, 255).to(torch.uint8)
    imageio.mimwrite(
        path, [frame.numpy() for frame in video_tensor], fps=fps, codec="libx264"
    )


def _add_frame_borders(canvas, num_context_frames, border_thickness=3):
    """Add colored borders to distinguish context (green) and prediction (red) frames.

    Args:
        canvas: torch.Tensor, shape [num_frames, C, H, W]
        num_context_frames: int, number of context frames
        border_thickness: int, thickness of the border in pixels
    """
    num_frames = canvas.shape[0]

    for frame_idx in range(num_frames):
        # Green border for context frames, red for prediction frames
        color = (0, 255, 0) if frame_idx < num_context_frames else (255, 0, 0)
        r, g, b = color

        H, W = canvas.shape[2], canvas.shape[3]

        # Top border
        canvas[frame_idx, 0, :border_thickness, :] = r
        canvas[frame_idx, 1, :border_thickness, :] = g
        canvas[frame_idx, 2, :border_thickness, :] = b

        # Bottom border
        canvas[frame_idx, 0, -border_thickness:, :] = r
        canvas[frame_idx, 1, -border_thickness:, :] = g
        canvas[frame_idx, 2, -border_thickness:, :] = b

        # Left border
        canvas[frame_idx, 0, :, :border_thickness] = r
        canvas[frame_idx, 1, :, :border_thickness] = g
        canvas[frame_idx, 2, :, :border_thickness] = b

        # Right border
        canvas[frame_idx, 0, :, -border_thickness:] = r
        canvas[frame_idx, 1, :, -border_thickness:] = g
        canvas[frame_idx, 2, :, -border_thickness:] = b

    return canvas


def _draw_box(frame, x1, y1, x2, y2, color, thickness=2, alpha=0.3):
    """Draw a colored rectangle with translucent fill on a [3, H, W] frame tensor in-place."""
    H, W = frame.shape[1:]
    x1 = max(0, min(W - 1, int(x1)))
    x2 = max(x1 + 1, min(W, int(x2)))
    y1 = max(0, min(H - 1, int(y1)))
    y2 = max(y1 + 1, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    thickness = max(1, thickness)

    # Color is RGB tuple (0-255)
    r, g, b = color

    # Draw translucent filled rectangle
    frame[0, y1:y2, x1:x2] = frame[0, y1:y2, x1:x2] * (1 - alpha) + r * alpha
    frame[1, y1:y2, x1:x2] = frame[1, y1:y2, x1:x2] * (1 - alpha) + g * alpha
    frame[2, y1:y2, x1:x2] = frame[2, y1:y2, x1:x2] * (1 - alpha) + b * alpha

    # Draw solid colored border
    frame[0, y1 : y1 + thickness, x1:x2] = r
    frame[1, y1 : y1 + thickness, x1:x2] = g
    frame[2, y1 : y1 + thickness, x1:x2] = b

    frame[0, y2 - thickness : y2, x1:x2] = r
    frame[1, y2 - thickness : y2, x1:x2] = g
    frame[2, y2 - thickness : y2, x1:x2] = b

    frame[0, y1:y2, x1 : x1 + thickness] = r
    frame[1, y1:y2, x1 : x1 + thickness] = g
    frame[2, y1:y2, x1 : x1 + thickness] = b

    frame[0, y1:y2, x2 - thickness : x2] = r
    frame[1, y1:y2, x2 - thickness : x2] = g
    frame[2, y1:y2, x2 - thickness : x2] = b


def _append_ground_truth_boxes_to_canvas(
    batch, batch_idx, canvas, num_context_frames, image_size
):
    """
    Draw ground truth bounding boxes on target frames only.

    Args:
        batch: dict containing 'instances' key
        batch_idx: int, index in batch
        canvas: [T, C, H, W] video tensor (T = num_context + num_target frames)
        num_context_frames: int, number of context frames (no ground truths for these)

    Returns:
        canvas with ground truth boxes drawn on target frames
    """
    if "instances" not in batch:
        return canvas

    # Per-class colors (same as detections)
    COLOR_MAP = {
        1: (255, 255, 0),  # yellow - TYPE_VEHICLE
        2: (255, 0, 0),  # red - TYPE_PEDESTRIAN
        3: (0, 255, 255),  # cyan - TYPE_CYCLIST
    }

    # Get instances for this batch item: List[Instances] of length T_target
    instances_list = batch["instances"][batch_idx]

    # Scaling to match model input size
    original_heights = batch["metadata"]["original_height"][batch_idx].cpu().tolist()
    original_widths = batch["metadata"]["original_width"][batch_idx].cpu().tolist()

    # Draw boxes on target frames (starting at num_context_frames)
    for t_target, instances in enumerate(instances_list):
        frame_idx = num_context_frames + t_target

        # Get boxes [N, 4] in [x1, y1, x2, y2] format and classes [N]
        boxes = instances.gt_boxes.tensor
        classes = instances.gt_classes

        scale_x = image_size / float(original_widths[frame_idx])
        scale_y = image_size / float(original_heights[frame_idx])

        # Draw each box
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box.tolist()
            x1 = round(x1 * scale_x)
            y1 = round(y1 * scale_y)
            x2 = round(x2 * scale_x)
            y2 = round(y2 * scale_y)
            color = COLOR_MAP.get(
                cls.item() + 1, (255, 255, 255)
            )  # white fallback. Convert back from model format {0,1,2} to COCO format {1,2,3}
            _draw_box(canvas[frame_idx], x1, y1, x2, y2, color, thickness=2, alpha=0.1)

    return canvas


def _append_detections_to_canvas(
    batch,
    batch_idx,
    canvas,
    image_size,
    model_outputs,
    detection_key="coco_predictions",
    score_thr=0.5,
    skip_context_frames=0,
):  # NEW PARAMETER
    # Per-class colors (RGB, 0-255)
    COLOR_MAP = {
        1: (255, 255, 0),  # yellow - TYPE_VEHICLE
        2: (255, 0, 0),  # red - TYPE_PEDESTRIAN
        3: (0, 255, 255),  # cyan - TYPE_CYCLIST
    }

    if (
        (model_outputs is not None)
        and (detection_key in model_outputs)
        and ("metadata" in batch)
    ):
        preds_by_image = defaultdict(list)
        for pred in model_outputs[detection_key]:
            if ("image_id" in pred) and ("bbox" in pred):
                # Apply score threshold filter
                if score_thr is not None and pred.get("score", 1.0) < score_thr:
                    continue

                # Store bbox, category_id, and score
                preds_by_image[int(pred["image_id"])].append(
                    {
                        "bbox": pred["bbox"],
                        "category_id": pred.get("category_id", -1),
                        "score": pred.get("score", 1.0),
                    }
                )

        image_ids = batch["metadata"]["image_ids"][batch_idx].cpu().tolist()
        original_heights = (
            batch["metadata"]["original_height"][batch_idx].cpu().tolist()
        )
        original_widths = batch["metadata"]["original_width"][batch_idx].cpu().tolist()

        for frame_idx, image_id in enumerate(image_ids):
            # NEW: Skip context frames if requested
            if frame_idx < skip_context_frames:
                continue

            detections = preds_by_image.get(int(image_id), [])
            if not detections:
                continue

            scale_x = image_size / float(original_widths[frame_idx])
            scale_y = image_size / float(original_heights[frame_idx])

            for det in detections:
                x, y, w, h = det["bbox"]
                x1 = round(x * scale_x)
                y1 = round(y * scale_y)
                x2 = round((x + w) * scale_x)
                y2 = round((y + h) * scale_y)

                # Get color based on category
                cat_id = det["category_id"]
                color = COLOR_MAP.get(cat_id, (255, 255, 255))  # white fallback

                _draw_box(
                    canvas[frame_idx], x1, y1, x2, y2, color, thickness=2, alpha=0.1
                )

    return canvas


def _append_batch_targets_to_canvas(batch, batch_idx, canvas, image_size):
    """Append target overlays to the canvas for a single batch item.
    Args:
        batch: dict, containing the batch data including targets
        batch_idx: int, index of the item in the batch
        canvas: torch.Tensor, shape [num_frames, C, H, W], the video canvas to append targets to
    """
    num_frames_strided, C, H, W = canvas.shape
    # Get the targets from the right batch index
    target_name_square = "target_square_heatmap_pixel"
    target_name_circle = "target_circle_heatmap_pixel"

    if (target_name_square not in batch) or (target_name_circle not in batch):
        return canvas
    # here we need to make a distinction between ndim == 5 (Deterministic and Flow models) and ndim ==4 (Heatmap models)
    if batch[target_name_square].ndim == 5:
        # Shape [B, num_frames_strided, H, W, 1]
        target_square = batch[target_name_square][batch_idx]  # [num_frames, H, W, 1]
        target_circle = batch[target_name_circle][batch_idx]  # [num_frames, H, W, 1]
    elif batch[target_name_square].ndim == 4:
        target_square = batch[target_name_square]  # Shape [num_frames, H, W, 1]
        target_circle = batch[target_name_circle]  # Shape [num_frames, H, W, 1]
    # Get video shape
    num_frames, C, H, W = canvas.shape
    # Create dark gray background for right side: [num_frames, C, H, W]
    dark_bg = torch.full(
        (num_frames_strided, C, image_size, image_size), 20, dtype=torch.uint8
    )  # [num_frames_strided, C, H, W]
    # Remove channel dimension from targets: [num_frames, H, W]
    sq_mask = target_square.squeeze(-1)
    cr_mask = target_circle.squeeze(-1)
    # Enlarge targets to 3x3 using max pooling with appropriate padding
    # Reshape to [1, num_frames, H, W] for conv operation
    sq_mask_batch = sq_mask.unsqueeze(0)
    cr_mask_batch = cr_mask.unsqueeze(0)
    # Use max_pool2d with kernel 3x3 and stride 1, then
    # pad back to original size
    sq_mask_large = torch.nn.functional.max_pool2d(
        sq_mask_batch, kernel_size=5, stride=1, padding=2
    )
    cr_mask_large = torch.nn.functional.max_pool2d(
        cr_mask_batch, kernel_size=5, stride=1, padding=2
    )
    # Remove batch dimension: [num_frames, H, W]
    sq_mask_large = sq_mask_large.squeeze(0)
    cr_mask_large = cr_mask_large.squeeze(0)

    # Overlay red for square (channel 0) and blue for circle (channel 2)
    dark_bg[:, 0, :, :][sq_mask_large > 0] = 255  # Red channel for square
    dark_bg[:, 2, :, :][cr_mask_large > 0] = 255  # Blue channel for circle
    # Concatenate along width: [num_frames, C, H, 2*W]
    canvas = torch.cat([canvas, dark_bg], dim=-1)
    # Add a white vertical line to separate the videos from the targets
    canvas[:, :, :, W] = 255
    return canvas


def _append_pca_latents_to_canvas(
    batch,
    model_outputs,
    batch_idx,
    canvas,
    image_size,
    patch_size,
    filter_background=True,
    background_threshold=0.5,
):
    # If one of the 2 keys is in ('context_latents', 'target_latents'), return canvas unchanged
    if ("context_latents" not in batch) or ("target_latents" not in batch):
        return canvas

    context_latents = batch["context_latents"][
        batch_idx
    ].cpu()  # [latent_dim, num_context_frames, num_h_patches, num_w_patches]
    target_latents = batch["target_latents"][
        batch_idx
    ].cpu()  # [latent_dim, num_target_frames, num_h_patches, num_w_patches]

    if model_outputs is not None:
        predicted_latents = model_outputs["predicted_latents"][
            batch_idx
        ].cpu()  # [latent_dim, num_target_frames, num_h_patches, num_w_patches]

    # Reshape back to [num_frames, 3, num_h_patches, num_w_patches]
    latent_dim, num_context_frames, num_h_patches, num_w_patches = context_latents.shape
    num_target_frames = target_latents.shape[1]

    # Flatten latents for PCA (fit on first context frame only)
    flattened_context_latents = rearrange(
        context_latents, "d t h w -> (t h w) d"
    )  # Shape: [num_context_frames * num_h_patches * num_w_patches, latent_dim]
    flattened_context_latents_fit = rearrange(
        context_latents[:, :1], "d t h w -> (t h w) d"
    )  # Shape: [num_h_patches * num_w_patches, latent_dim]
    flattened_target_latents = rearrange(
        target_latents, "d t h w -> (t h w) d"
    )  # Shape: [num_target_frames * num_h_patches * num_w_patches, latent_dim]
    if model_outputs is not None:
        flattened_predicted_latents = rearrange(
            predicted_latents, "d t h w -> (t h w) d"
        )  # Shape: [num_target_frames * num_h_patches * num_w_patches, latent_dim]

    if filter_background:
        pca_mask = PCA(n_components=1)
        pca_mask.fit(flattened_context_latents_fit.numpy())
        pca_mask_flattened_context = pca_mask.transform(
            flattened_context_latents.numpy()
        )
        pca_mask_flattened_target = pca_mask.transform(flattened_target_latents.numpy())
        if model_outputs is not None:
            pca_mask_flattened_predicted = pca_mask.transform(
                flattened_predicted_latents.numpy()
            )

        mask_context = rearrange(
            torch.tensor(pca_mask_flattened_context),
            "(t h w) c -> t c h w",
            t=num_context_frames,
            h=num_h_patches,
            w=num_w_patches,
        )
        mask_target = rearrange(
            torch.tensor(pca_mask_flattened_target),
            "(t h w) c -> t c h w",
            t=num_target_frames,
            h=num_h_patches,
            w=num_w_patches,
        )
        if model_outputs is not None:
            mask_predicted = rearrange(
                torch.tensor(pca_mask_flattened_predicted),
                "(t h w) c -> t c h w",
                t=num_target_frames,
                h=num_h_patches,
                w=num_w_patches,
            )

        mask_context = (
            torch.sigmoid(0.8 * mask_context) > background_threshold
        ).float()
        mask_target = (torch.sigmoid(0.8 * mask_target) > background_threshold).float()
        if model_outputs is not None:
            mask_predicted = (
                torch.sigmoid(0.8 * mask_predicted) > background_threshold
            ).float()

    # Fit PCA on first context frame latents, apply to all frames
    pca = PCA(n_components=3)
    pca.fit(flattened_context_latents_fit.numpy())
    pca_flattened_context_latents = pca.transform(
        flattened_context_latents.numpy()
    )  # [num_context_frames * num_h_patches * num_w_patches, 3]
    pca_flattened_target_latents = pca.transform(flattened_target_latents.numpy())
    if model_outputs is not None:
        pca_flattened_predicted_latents = pca.transform(
            flattened_predicted_latents.numpy()
        )

    pca_context_latents = rearrange(
        torch.tensor(pca_flattened_context_latents),
        "(t h w) c -> t c h w",
        t=num_context_frames,
        h=num_h_patches,
        w=num_w_patches,
    )  # Shape: [num_context_frames, 3, num_h_patches, num_w_patches]

    pca_target_latents = rearrange(
        torch.tensor(pca_flattened_target_latents),
        "(t h w) c -> t c h w",
        t=num_target_frames,
        h=num_h_patches,
        w=num_w_patches,
    )  # Shape: [num_target_frames, 3, num_h_patches, num_w_patches]

    if filter_background:
        pca_context_latents = pca_context_latents * mask_context
        pca_target_latents = pca_target_latents * mask_target

    if model_outputs is not None:
        pca_predicted_latents = rearrange(
            torch.tensor(pca_flattened_predicted_latents),
            "(t h w) c -> t c h w",
            t=num_target_frames,
            h=num_h_patches,
            w=num_w_patches,
        )  # Shape: [num_target_frames, 3, num_h_patches, num_w_patches]
        if filter_background:
            pca_predicted_latents = pca_predicted_latents * mask_predicted

    # Create 2 tensors: context + target latents, context + predicted latents
    pca_context_and_target = torch.cat(
        [pca_context_latents, pca_target_latents], dim=0
    )  # Shape: [num_context_frames + num_target_frames, 3, num_h_patches, num_w_patches]
    if model_outputs is not None:
        pca_context_and_predicted = torch.cat(
            [pca_context_latents, pca_predicted_latents], dim=0
        )  # Shape: [num_context_frames + num_target_frames, 3, num_h_patches, num_w_patches]

    # Upsample latents to image size, but repeat the same value per patch. Patch dimension is 16
    pca_context_and_target_upsamples = pca_context_and_target.repeat_interleave(
        patch_size, dim=2
    ).repeat_interleave(patch_size, dim=3)  # Shape: [num_frames, 3, H, W]

    if model_outputs is not None:
        pca_context_and_predicted_upsamples = (
            pca_context_and_predicted.repeat_interleave(
                patch_size, dim=2
            ).repeat_interleave(patch_size, dim=3)
        )  # Shape: [num_frames, 3, H, W]

    # Apply sigmoid (2 * x)
    pca_context_and_target_upsamples = torch.sigmoid(
        0.8 * pca_context_and_target_upsamples
    )

    if model_outputs is not None:
        pca_context_and_predicted_upsamples = torch.sigmoid(
            0.8 * pca_context_and_predicted_upsamples
        )

    # Scale to [0,255]
    pca_context_and_target_upsamples = (
        (pca_context_and_target_upsamples * 255).clamp(0, 255).byte()
    )

    if model_outputs is not None:
        pca_context_and_predicted_upsamples = (
            (pca_context_and_predicted_upsamples * 255).clamp(0, 255).byte()
        )

    # Append to canvas side by side with the white line separator
    # First append context + target latents
    canvas = torch.cat(
        [canvas, pca_context_and_target_upsamples], dim=-1
    )  # [num_frames, C, H, 3*W]
    canvas[:, :, :, image_size] = 255  # White line separator
    next_separator = 2 * image_size

    if model_outputs is not None:
        # Then append context + predicted latents
        canvas = torch.cat(
            [canvas, pca_context_and_predicted_upsamples], dim=-1
        )  # [num_frames, C, H, 4*W]
        canvas[:, :, :, next_separator] = 255  # White line separator
    return canvas  # [num_frames, 3, H, 3*W] (4*W if predicted latents are appended)


def _append_model_predictions_to_canvas(
    canvas, model_outputs, batch_idx, image_size, prediction_type
):
    """
    Append oracle and predicted heatmaps to canvas (context+oracle, context+predictions).
    The outputs are of shape [num_frames, C, H, W] and have binary targets

    """
    if model_outputs is None:
        return canvas
    elif "context_square_heatmap_pixel" not in model_outputs:
        return canvas
    elif prediction_type == "latents":
        num_frames, C, H, W = canvas.shape

        # Get heatmaps: [num_ctx/tgt, H, W]
        ctx_sq = model_outputs["context_square_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_ctx, img_size, img_size]
        ctx_cr = model_outputs["context_circle_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_ctx, img_size, img_size]
        oracle_sq = model_outputs["oracle_square_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]
        oracle_cr = model_outputs["oracle_circle_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]
        pred_sq = model_outputs["predicted_square_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]
        pred_cr = model_outputs["predicted_circle_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]

        # Also get denormalized versions for context and oracle
        ctx_sq_denorm = model_outputs["context_denormalized_square_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_ctx, img_size, img_size]
        ctx_cr_denorm = model_outputs["context_denormalized_circle_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_ctx, img_size, img_size]
        oracle_sq_denorm = model_outputs["oracle_denormalized_square_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]
        oracle_cr_denorm = model_outputs["oracle_denormalized_circle_heatmap_pixel"][
            batch_idx
        ].cpu()  # [num_tgt, img_size, img_size]

        # Convert context to binary using argmax over spatial dimensions
        ctx_sq_flat = ctx_sq.flatten(1)  # [num_ctx, H*W]
        ctx_sq_argmax = ctx_sq_flat.argmax(dim=1)  # [num_ctx]
        ctx_sq_binary = torch.zeros_like(ctx_sq_flat)
        ctx_sq_binary.scatter_(1, ctx_sq_argmax.unsqueeze(1), 1.0)
        ctx_sq_binary = ctx_sq_binary.view_as(ctx_sq)  # [num_ctx, H, W]

        ctx_cr_flat = ctx_cr.flatten(1)
        ctx_cr_argmax = ctx_cr_flat.argmax(dim=1)
        ctx_cr_binary = torch.zeros_like(ctx_cr_flat)
        ctx_cr_binary.scatter_(1, ctx_cr_argmax.unsqueeze(1), 1.0)
        ctx_cr_binary = ctx_cr_binary.view_as(ctx_cr)

        ctx_sq_denorm_flat = ctx_sq_denorm.flatten(1)  # [num_ctx, H*W]
        ctx_sq_denorm_argmax = ctx_sq_denorm_flat.argmax(dim=1)  # [num_ctx]
        ctx_sq_denorm_binary = torch.zeros_like(ctx_sq_denorm_flat)
        ctx_sq_denorm_binary.scatter_(1, ctx_sq_denorm_argmax.unsqueeze(1), 1.0)
        ctx_sq_denorm_binary = ctx_sq_denorm_binary.view_as(
            ctx_sq_denorm
        )  # [num_ctx, H, W]

        ctx_cr_denorm_flat = ctx_cr_denorm.flatten(1)
        ctx_cr_denorm_argmax = ctx_cr_denorm_flat.argmax(dim=1)
        ctx_cr_denorm_binary = torch.zeros_like(ctx_cr_denorm_flat)
        ctx_cr_denorm_binary.scatter_(1, ctx_cr_denorm_argmax.unsqueeze(1), 1.0)
        ctx_cr_denorm_binary = ctx_cr_denorm_binary.view_as(
            ctx_cr_denorm
        )  # [num_ctx, H, W]

        # Convert predictions to binary using argmax over spatial dimensions
        pred_sq_flat = pred_sq.flatten(1)  # [num_tgt, H*W]
        pred_sq_argmax = pred_sq_flat.argmax(dim=1)  # [num_tgt]
        pred_sq_binary = torch.zeros_like(pred_sq_flat)
        pred_sq_binary.scatter_(1, pred_sq_argmax.unsqueeze(1), 1.0)
        pred_sq_binary = pred_sq_binary.view_as(pred_sq)  # [num_tgt, H, W]

        pred_cr_flat = pred_cr.flatten(1)
        pred_cr_argmax = pred_cr_flat.argmax(dim=1)
        pred_cr_binary = torch.zeros_like(pred_cr_flat)
        pred_cr_binary.scatter_(1, pred_cr_argmax.unsqueeze(1), 1.0)
        pred_cr_binary = pred_cr_binary.view_as(pred_cr)

        # Convert oracle to binary using argmax over spatial dimensions
        oracle_sq_flat = oracle_sq.flatten(1)  # [num_tgt, H*W]
        oracle_sq_argmax = oracle_sq_flat.argmax(dim=1)  # [num_tgt]
        oracle_sq_binary = torch.zeros_like(oracle_sq_flat)
        oracle_sq_binary.scatter_(1, oracle_sq_argmax.unsqueeze(1), 1.0)
        oracle_sq_binary = oracle_sq_binary.view_as(oracle_sq)  # [num_tgt, H, W]

        oracle_cr_flat = oracle_cr.flatten(1)
        oracle_cr_argmax = oracle_cr_flat.argmax(dim=1)
        oracle_cr_binary = torch.zeros_like(oracle_cr_flat)
        oracle_cr_binary.scatter_(1, oracle_cr_argmax.unsqueeze(1), 1.0)
        oracle_cr_binary = oracle_cr_binary.view_as(oracle_cr)

        oracle_sq_denorm_flat = oracle_sq_denorm.flatten(1)  # [num_tgt, H*W]
        oracle_sq_denorm_argmax = oracle_sq_denorm_flat.argmax(dim=1)  # [num_tgt]
        oracle_sq_denorm_binary = torch.zeros_like(oracle_sq_denorm_flat)
        oracle_sq_denorm_binary.scatter_(1, oracle_sq_denorm_argmax.unsqueeze(1), 1.0)
        oracle_sq_denorm_binary = oracle_sq_denorm_binary.view_as(
            oracle_sq_denorm
        )  # [num_tgt, H, W]

        oracle_cr_denorm_flat = oracle_cr_denorm.flatten(1)
        oracle_cr_denorm_argmax = oracle_cr_denorm_flat.argmax(dim=1)
        oracle_cr_denorm_binary = torch.zeros_like(oracle_cr_denorm_flat)
        oracle_cr_denorm_binary.scatter_(1, oracle_cr_denorm_argmax.unsqueeze(1), 1.0)
        oracle_cr_denorm_binary = oracle_cr_denorm_binary.view_as(
            oracle_cr_denorm
        )  # [num_tgt, H, W]

        # Concatenate context + oracle/predictions: [num_frames, H, W]
        ctx_oracle_sq = torch.cat([ctx_sq_binary, oracle_sq_binary], dim=0)
        ctx_oracle_cr = torch.cat([ctx_cr_binary, oracle_cr_binary], dim=0)

        ctx_oracle_sq_denorm = torch.cat(
            [ctx_sq_denorm_binary, oracle_sq_denorm_binary], dim=0
        )
        ctx_oracle_cr_denorm = torch.cat(
            [ctx_cr_denorm_binary, oracle_cr_denorm_binary], dim=0
        )

        ctx_pred_sq = torch.cat([ctx_sq_binary, pred_sq_binary], dim=0)
        ctx_pred_cr = torch.cat([ctx_cr_binary, pred_cr_binary], dim=0)

        # Enlarge with 5x5 max pooling: [1, num_frames, H, W] -> [num_frames, H, W]
        oracle_sq_large = torch.nn.functional.max_pool2d(
            ctx_oracle_sq.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        oracle_cr_large = torch.nn.functional.max_pool2d(
            ctx_oracle_cr.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        oracle_sq_denorm_large = torch.nn.functional.max_pool2d(
            ctx_oracle_sq_denorm.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        oracle_cr_denorm_large = torch.nn.functional.max_pool2d(
            ctx_oracle_cr_denorm.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        pred_sq_large = torch.nn.functional.max_pool2d(
            ctx_pred_sq.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        pred_cr_large = torch.nn.functional.max_pool2d(
            ctx_pred_cr.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)

        # Create dark backgrounds and overlay: [num_frames, C, H, W]
        oracle_bg = torch.full((num_frames, C, H, image_size), 20, dtype=torch.uint8)
        oracle_denorm_bg = torch.full(
            (num_frames, C, H, image_size), 20, dtype=torch.uint8
        )
        pred_bg = torch.full((num_frames, C, H, image_size), 20, dtype=torch.uint8)

        oracle_bg[:, 0, :, :][oracle_sq_large > 0] = 255  # Red for square
        oracle_bg[:, 2, :, :][oracle_cr_large > 0] = 255  # Blue for circle
        oracle_denorm_bg[:, 0, :, :][oracle_sq_denorm_large > 0] = 255
        oracle_denorm_bg[:, 2, :, :][oracle_cr_denorm_large > 0] = 255
        pred_bg[:, 0, :, :][pred_sq_large > 0] = 255
        pred_bg[:, 2, :, :][pred_cr_large > 0] = 255

        # Concatenate with white separators
        canvas = torch.cat([canvas, oracle_bg], dim=-1)
        canvas[:, :, :, -image_size - 1] = 255

        canvas = torch.cat([canvas, oracle_denorm_bg], dim=-1)
        canvas[:, :, :, -image_size - 1] = 255

        canvas = torch.cat([canvas, pred_bg], dim=-1)
        canvas[:, :, :, -image_size - 1] = 255

    elif prediction_type == "heatmaps":
        num_frames, C, H, W = canvas.shape
        circle_heatmap_pixel = model_outputs[
            "circle_heatmap_pixel"
        ].cpu()  # [num_frames, 1, H, W]
        square_heatmap_pixel = model_outputs[
            "square_heatmap_pixel"
        ].cpu()  # [num_frames, 1, H, W]

        square_binary = (
            (
                square_heatmap_pixel.flatten(2)
                == square_heatmap_pixel.flatten(2).max(dim=2, keepdim=True)[0]
            )
            .float()
            .view_as(square_heatmap_pixel)
        )
        circle_binary = (
            (
                circle_heatmap_pixel.flatten(2)
                == circle_heatmap_pixel.flatten(2).max(dim=2, keepdim=True)[0]
            )
            .float()
            .view_as(circle_heatmap_pixel)
        )

        square_large = torch.nn.functional.max_pool2d(
            square_binary.squeeze(1).unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)
        circle_large = torch.nn.functional.max_pool2d(
            circle_binary.squeeze(1).unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)

        pred_bg = torch.full((num_frames, C, H, image_size), 20, dtype=torch.uint8)
        pred_bg[:, 0, :, :][square_large > 0] = 255
        pred_bg[:, 2, :, :][circle_large > 0] = 255

        canvas = torch.cat([canvas, pred_bg], dim=-1)
        canvas[:, :, :, -image_size - 1] = 255

    return canvas


def save_canvas_as_video(
    canvas,
    gpu_rank,
    batch_idx,
    split,
    dump_epoch_dir,
    dataloader_batch_idx=None,
    prediction_id=0,
    image_size=512,
    num_context_frames=None,
    context_frames_to_show=4,
    target_frames_to_show=12,
    file_prefix="",
    fps=10,
):
    """Save the canvas as a video file and horizontal strips per video section.
    Args:
        canvas: torch.Tensor, shape [num_frames, C, H, W], the video canvas to save
        gpu_rank: int, GPU rank for distributed training
        batch_idx: int, index of the item in the batch
        split: str, 'training' or 'evaluation'
        dump_epoch_dir: str, directory to dump the epoch results
        dataloader_batch_idx: int, dataloader batch index
        prediction_id: int, identifier for multiple predictions
        image_size: int, width of each video section
        num_context_frames: int or None, number of context frames in the sequence
        context_frames_to_show: int, number of context frames to keep (last ones)
        target_frames_to_show: int, number of target frames to keep (first ones)
    """
    # first check that split is either 'training' or 'evaluation'
    if split not in {"training", "evaluation"}:
        raise ValueError(
            f"Unsupported split '{split}'. Expected 'training' or 'evaluation'."
        )

    logger = logging.getLogger(__name__)
    split_dir = os.path.join(dump_epoch_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Create base filename (without extension)
    base_filename = f"{file_prefix}gpu_{gpu_rank}_dataloader_{dataloader_batch_idx}_batch_{batch_idx:04d}_pred_{prediction_id}"

    # Video path
    video_path = os.path.join(split_dir, f"{base_filename}.mp4")

    num_frames, C, H, W = canvas.shape

    # Folder for horizontal strips
    frames_dir = os.path.join(split_dir, base_filename)
    os.makedirs(frames_dir, exist_ok=True)

    # === NEW: Split canvas into video sections and save horizontal strips ===
    # Calculate number of video sections (assuming each section is image_size wide)
    num_sections = W // image_size

    # Section names for better organization
    section_names = [
        "original_video",
        "oracle_predictions",
        "model_predictions",
        "ground_truth_latents",
        "predicted_latents",
    ]

    if num_context_frames is None:
        selected_indices = list(range(num_frames))
    else:
        context_frames_to_show = max(0, min(num_context_frames, context_frames_to_show))
        target_total = max(0, num_frames - num_context_frames)
        target_frames_to_show = max(0, min(target_total, target_frames_to_show))
        context_start = num_context_frames - context_frames_to_show
        selected_indices = list(range(context_start, num_context_frames))
        selected_indices += list(
            range(num_context_frames, num_context_frames + target_frames_to_show)
        )

    num_sections = min(num_sections, len(section_names))
    for section_idx in range(num_sections):
        # Extract this section: [num_frames, C, H, image_size]
        start_col = section_idx * image_size
        end_col = start_col + image_size
        section_canvas = canvas[
            :, :, :, start_col:end_col
        ]  # [num_frames, C, H, image_size]

        # Stack all frames horizontally: [C, H, num_frames * image_size]
        horizontal_strip = torch.cat(
            [section_canvas[i] for i in selected_indices], dim=-1
        )

        # Save this horizontal strip
        section_name = (
            section_names[section_idx]
            if section_idx < len(section_names)
            else f"section_{section_idx}"
        )
        strip_path = os.path.join(frames_dir, f"canvas_{section_name}.png")
        write_png(horizontal_strip.to(torch.uint8), strip_path)

        logger.info(f"Saved horizontal strip for {section_name} to {strip_path}")

    # Rearrange canvas from shape [num_frames, C, H, W] to [num_frames, H, W, C] for video
    canvas_video = rearrange(canvas, "t c h w -> t h w c")

    # Write video using torchvision
    write_video(video_path, canvas_video, fps=fps)
    logger.info(f"Saved visualization video to {video_path}")


def _compose_decoded_future_canvas(
    original_video,
    decoded_future_video,
    num_context_frames,
):
    canvas = original_video.clone()
    canvas[num_context_frames:] = decoded_future_video
    return _add_frame_borders(canvas, num_context_frames, border_thickness=3)


def _save_decoded_video_comparisons(
    batch,
    decoded_oracle_videos,
    decoded_predicted_videos,
    *,
    dataloader_batch_idx,
    gpu_rank,
    split,
    dump_epoch_dir,
    num_context_frames,
    prediction_id=0,
    fps=10,
):
    video = batch["video"].cpu()
    batch_size = video.shape[0]
    image_size = video.shape[-1]

    for batch_idx in range(batch_size):
        original_canvas = _add_frame_borders(
            video[batch_idx].clone(),
            num_context_frames,
            border_thickness=3,
        )
        oracle_canvas = _compose_decoded_future_canvas(
            video[batch_idx],
            decoded_oracle_videos[batch_idx].cpu(),
            num_context_frames,
        )
        predicted_canvas = _compose_decoded_future_canvas(
            video[batch_idx],
            decoded_predicted_videos[batch_idx].cpu(),
            num_context_frames,
        )

        canvas = torch.cat(
            [original_canvas, oracle_canvas, predicted_canvas],
            dim=-1,
        )
        for panel_idx in range(1, 3):
            canvas[:, :, :, panel_idx * image_size] = 255

        save_canvas_as_video(
            canvas=canvas,
            dataloader_batch_idx=dataloader_batch_idx,
            gpu_rank=gpu_rank,
            batch_idx=batch_idx,
            split=split,
            dump_epoch_dir=dump_epoch_dir,
            prediction_id=prediction_id,
            image_size=image_size,
            num_context_frames=num_context_frames,
            context_frames_to_show=4,
            target_frames_to_show=12,
            file_prefix="decoded_",
            fps=fps,
        )


def save_batch_and_model_outputs(
    batch,
    dataloader_batch_idx,
    model_outputs,
    gpu_rank,
    prediction_type,
    split,
    dump_epoch_dir,
    num_context_frames,
    prediction_id=0,
    visualize_ground_truths=False,
    patch_size=None,
    filter_background=True,
    background_threshold=0.5,
    file_prefix="",
):
    if batch is None and model_outputs is None:
        raise ValueError("Both batch and model_outputs are None. Nothing to save.")

    video = batch["video"].cpu()
    batch_size = video.shape[0]
    image_size = video.shape[-1]

    for batch_idx in range(batch_size):
        canvases = []

        # === Ground Truth Canvas (optional, leftmost) ===
        if visualize_ground_truths and "instances" in batch:
            gt_canvas = video[batch_idx].clone()
            gt_canvas = _add_frame_borders(
                gt_canvas, num_context_frames, border_thickness=4
            )
            canvases.append(gt_canvas)

        # === Oracle Canvas ===
        oracle_canvas = video[batch_idx].clone()
        oracle_canvas = _add_frame_borders(
            oracle_canvas, num_context_frames, border_thickness=3
        )
        oracle_canvas = _append_detections_to_canvas(
            batch=batch,
            batch_idx=batch_idx,
            canvas=oracle_canvas,
            image_size=image_size,
            model_outputs=model_outputs,
            detection_key="coco_detections_oracle",
        )
        canvases.append(oracle_canvas)

        # === Model Canvas ===
        model_canvas = video[batch_idx].clone()
        # Convert to grayscale for both context and future frames
        gray = (
            0.299 * model_canvas[:, 0]
            + 0.587 * model_canvas[:, 1]
            + 0.114 * model_canvas[:, 2]
        )
        gray = gray * 0.3  # Darken: 0.0 = black, 1.0 = original brightness
        model_canvas = torch.stack(
            [gray, gray, gray], dim=1
        )  # [T, 3, H, W] - grayscale in 3 channels
        model_canvas = model_canvas.to(torch.uint8)
        model_canvas = _add_frame_borders(
            model_canvas, num_context_frames, border_thickness=3
        )
        model_canvas = _append_detections_to_canvas(
            batch=batch,
            batch_idx=batch_idx,
            canvas=model_canvas,
            image_size=image_size,
            model_outputs=model_outputs,
            detection_key="coco_detections",
            skip_context_frames=num_context_frames,
        )
        canvases.append(model_canvas)

        # === Concatenate all canvases with white separators ===
        canvas = torch.cat(canvases, dim=-1)  # [T, C, H, W*num_canvases]
        # Add white separators between canvases
        for i in range(1, len(canvases)):
            canvas[:, :, :, i * image_size] = 255

        # === Append additional visualizations ===
        split_dir = os.path.join(dump_epoch_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        base_filename = f"{file_prefix}gpu_{gpu_rank}_dataloader_{dataloader_batch_idx}_batch_{batch_idx:04d}"
        if prediction_id == 0 and "context_latents" in batch:
            context_latents = batch["context_latents"][batch_idx].cpu()
            torch.save(
                context_latents,
                os.path.join(split_dir, f"{base_filename}_context_latents.pt"),
            )
        if model_outputs is not None:
            if "predicted_latents" in model_outputs:
                predicted_latents = model_outputs["predicted_latents"][batch_idx].cpu()
            else:
                predicted_latents = None
            if predicted_latents is not None:
                torch.save(
                    predicted_latents,
                    os.path.join(
                        split_dir,
                        f"{base_filename}_predicted_latents_pred_{prediction_id}.pt",
                    ),
                )

        canvas = _append_pca_latents_to_canvas(
            batch,
            model_outputs,
            batch_idx,
            canvas,
            image_size,
            patch_size=patch_size,
            filter_background=filter_background,
            background_threshold=background_threshold,
        )
        canvas = _append_batch_targets_to_canvas(batch, batch_idx, canvas, image_size)
        canvas = _append_model_predictions_to_canvas(
            canvas, model_outputs, batch_idx, image_size, prediction_type
        )

        save_canvas_as_video(
            canvas=canvas,
            dataloader_batch_idx=dataloader_batch_idx,
            gpu_rank=gpu_rank,
            batch_idx=batch_idx,
            split=split,
            dump_epoch_dir=dump_epoch_dir,
            prediction_id=prediction_id,
            image_size=image_size,
            num_context_frames=num_context_frames,
            context_frames_to_show=4,
            target_frames_to_show=12,
            file_prefix=file_prefix,
        )
