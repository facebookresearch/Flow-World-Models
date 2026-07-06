import os
import logging

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from einops import rearrange
from torch.utils.data import Dataset
import pandas as pd


class WaymoVideoDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.logger = logging.getLogger(__name__)
        self.root = cfg.root
        df_metadata_path = os.path.join(self.root, "df_metadata.csv")
        self.df_metadata = pd.read_csv(df_metadata_path)
        self.number_context_latents = cfg.number_context_latents
        self.number_target_latents = cfg.number_target_latents
        self.number_total_latents = (
            self.number_context_latents + self.number_target_latents
        )
        self.image_size = cfg.image_size
        self.resize = cfg.resize
        self.horizontal_flip = cfg.horizontal_flip
        self.reverse_flip = cfg.reverse_flip
        self.prob_flip = 0.5

        # Build mappings for (image_id -> clip_id) and (clip_id -> image_ids)
        self.clip_id_to_image_ids = self._build_clip_id_to_image_id_map()
        self.image_id_to_clip_id = self._build_image_id_to_clip_id_map()

        # Build full and minival image ID lists
        if split == "val_mini":
            self.val_img_ids_list, self.val_clip_ids_list = (
                self._build_minival_clip_and_img_indexes()
            )
            before_len = len(self.df_metadata)
            before_clip_count = self.df_metadata["clip_id"].nunique()
            self.df_metadata = self.df_metadata[
                self.df_metadata["clip_id"].isin(self.val_clip_ids_list)
            ].reset_index(drop=True)
            after_len = len(self.df_metadata)
            after_clip_count = self.df_metadata["clip_id"].nunique()
            # assert after_len == after_clip_count
            assert after_len == after_clip_count, (
                f"Expected number of rows {after_clip_count}, got {after_len}"
            )
            remaining_pct = after_clip_count / before_clip_count * 100.0
            self.logger.info(
                "val_mini df_metadata rows %d -> %d; clip_ids %d -> %d (%.2f%% remaining)",
                before_len,
                after_len,
                before_clip_count,
                after_clip_count,
                remaining_pct,
            )
        elif split == "val":
            self.val_img_ids_list, self.val_clip_ids_list = (
                self._build_full_clip_and_img_indexes()
            )
            before_len = len(self.df_metadata)
            before_clip_count = self.df_metadata["clip_id"].nunique()
            self.df_metadata = self.df_metadata[
                self.df_metadata["clip_id"].isin(self.val_clip_ids_list)
            ].reset_index(drop=True)
            after_len = len(self.df_metadata)
            after_clip_count = self.df_metadata["clip_id"].nunique()
            remaining_pct = after_clip_count / before_clip_count * 100.0
            self.logger.info(
                "val df_metadata rows %d -> %d; clip_ids %d -> %d (%.2f%% remaining)",
                before_len,
                after_len,
                before_clip_count,
                after_clip_count,
                remaining_pct,
            )

    def _build_clip_id_to_image_id_map(self):
        """Build mapping from clip index to image_ids."""
        clip_map = {}
        for idx in range(len(self.df_metadata)):
            clip_id = self.df_metadata.iloc[idx]["clip_id"]
            image_ids_str = self.df_metadata.iloc[idx]["image_ids"]
            image_ids = list(map(int, image_ids_str.strip("[]").split(",")))
            clip_map[clip_id] = image_ids
        return clip_map

    def _build_image_id_to_clip_id_map(self):
        """Build mapping from image_id to clip index."""
        image_to_clip = {}
        for clip_id, image_ids in self.clip_id_to_image_ids.items():
            for img_id in image_ids:
                image_to_clip[img_id] = clip_id
        return image_to_clip

    def _build_minival_clip_and_img_indexes(self):
        """
        Build list of image IDs and clip IDs from first subclip of each parent video.
        Also filters self.df_metadata to keep only these clips.
        """
        first_clips_per_video = (
            self.df_metadata.groupby(["parent_video_id", "camera_name"])
            .first()
            .reset_index()
        )

        minival_clip_ids = []
        minival_img_ids = []

        for idx in range(len(first_clips_per_video)):
            clip_id = first_clips_per_video.iloc[idx]["clip_id"]
            image_ids = self.get_image_ids_from_clip_id(clip_id, frame_type="target")
            minival_clip_ids.append(clip_id)
            minival_img_ids.extend(image_ids)

        # Filter the dataframe to keep only the minival clips

        self.logger.info(
            f"Minival: {len(minival_clip_ids)} clips, {len(minival_img_ids)} target images"
        )

        return sorted(minival_img_ids), sorted(minival_clip_ids)

    def _build_full_clip_and_img_indexes(self):
        """Build list of all image IDs and clip IDs from all clips."""
        full_clip_ids = []
        full_img_ids = []

        for idx in range(len(self.df_metadata)):
            clip_id = self.df_metadata.iloc[idx]["clip_id"]
            image_ids = self.get_image_ids_from_clip_id(clip_id, frame_type="target")
            full_clip_ids.append(clip_id)
            full_img_ids.extend(image_ids)

        self.logger.info(
            f"Full val: {len(full_clip_ids)} clips, {len(full_img_ids)} target images"
        )

        return sorted(full_img_ids), sorted(full_clip_ids)

    def get_clip_id_from_image_id(self, image_id):
        """
        Get clip index for a given image_id.

        Args:
            image_id: COCO image ID

        Returns:
            clip_id (int) or None if not found
        """
        return self.image_id_to_clip_id[image_id]

    def get_image_ids_from_clip_id(self, clip_id, frame_type=None):
        """
        Get image IDs for a clip.

        Args:
            clip_id: Unique clip ID from the dataframe 'clip_id' column
            frame_type: 'context', 'target', or None for all frames

        Returns:
            list of image_ids
        """
        # Find the dataframe index for this clip_id
        image_ids = self.clip_id_to_image_ids[clip_id]

        if frame_type == "context":
            res = image_ids[: self.number_context_latents]
            assert len(res) == self.number_context_latents, (
                f"Expected {self.number_context_latents} context image IDs, got {len(res)}"
            )
            return res
        elif frame_type == "target":
            res = image_ids[self.number_context_latents :]
            assert len(res) == self.number_target_latents, (
                f"Expected {self.number_target_latents} target image IDs, got {len(res)}"
            )
            return res
        else:  # None
            return image_ids

    def __len__(self):
        return len(self.df_metadata)

    def __getitem__(self, idx):
        row = self.df_metadata.iloc[idx]
        video_file_name = row["video_filename"]

        pth_video_tensor_pt = os.path.join(self.root, video_file_name)
        metadata = self.df_metadata.iloc[idx].to_dict()

        # --- image_ids ---
        image_ids = list(map(int, metadata["image_ids"].strip("[]").split(",")))
        image_ids = torch.tensor(image_ids, dtype=torch.long)
        metadata["image_ids"] = image_ids

        # --- original height / width ---
        original_height = list(map(int, row["original_height"].strip("[]").split(",")))
        original_width = list(map(int, row["original_width"].strip("[]").split(",")))

        # convert to tensors
        original_height = torch.tensor(original_height, dtype=torch.long)  # shape [T]
        original_width = torch.tensor(original_width, dtype=torch.long)  # shape [T]

        metadata["original_height"] = original_height
        metadata["original_width"] = original_width

        video_tensor = torch.load(pth_video_tensor_pt)  # expect shape (T, C, H, W)

        if self.horizontal_flip and torch.rand(1) < self.prob_flip:
            video_tensor = torch.flip(video_tensor, dims=[3])  # flip width dimension
            metadata["horizontal_flip"] = True
        else:
            metadata["horizontal_flip"] = False

        if self.reverse_flip and torch.rand(1) < self.prob_flip:
            video_tensor = torch.flip(video_tensor, dims=[0])  # flip time dimension
            metadata["time_reversed"] = True
        else:
            metadata["time_reversed"] = False

        if metadata["time_reversed"]:
            image_ids = torch.flip(image_ids, dims=[0])
            original_height = torch.flip(original_height, dims=[0])
            original_width = torch.flip(original_width, dims=[0])
            metadata["image_ids"] = image_ids
            metadata["original_height"] = original_height
            metadata["original_width"] = original_width

        # check that the shape is correct
        video_tensor_shape = video_tensor.shape
        if self.resize:
            video_tensor = self._resize_video(
                video_tensor, self.image_size, method="interpolate"
            )
        # video tensor must be in [T, C, H, W]
        assert len(video_tensor_shape) == 4, (
            f"Expected video tensor to have 4 dimensions, got {len(video_tensor_shape)}"
        )
        assert video_tensor_shape[0] == self.number_total_latents, (
            f"Expected video tensor to have {self.number_total_latents} frames, got {video_tensor_shape[0]}"
        )
        assert video_tensor_shape[1] == 3, (
            f"Expected video tensor to have 3 channels, got {video_tensor_shape[1]}"
        )
        assert video_tensor_shape[3] == video_tensor_shape[2], (
            f"Expected video tensor to have shape {self.image_size}x{self.image_size}, got {video_tensor_shape[2]}x{video_tensor_shape[3]}"
        )

        data = dict(
            video=video_tensor,
            metadata=metadata,
        )

        return data

    @staticmethod
    def _resize_video(video_tensor, img_size, method):
        video_tensor = rearrange(video_tensor, "T H W C -> T C H W")
        if method == "resize":
            resized = tvF.resize(video_tensor, [img_size, img_size])
        else:
            resized = F.interpolate(
                video_tensor,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        return resized
