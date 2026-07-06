# Waymo → WaymoCOCO conversion

`convert_waymo_to_coco.py` turns raw **Waymo Open Dataset v1.4.3** `.tfrecord` segments
into the WaymoCOCO layout the training pipeline consumes: per-clip video tensors
(`*_video_tensors/{clip_id}.pt`, `[16, 3, 512, 512]` uint8), COCO detection annotations
(`annotations/instances_{train,val}2020.json`; 3 classes: vehicle / pedestrian / cyclist),
and a `df_metadata.csv` linking clips to frames.

## Isolated environment

This converter runs in **its own uv environment**, separate from the root FlowWM project.
It needs TensorFlow + `protobuf<4` (to read the tfrecords and the vendored Waymo protos),
which conflicts with the modern protobuf the training stack resolves to. Keeping it here
means `uv sync` at the repo root stays lean (no TensorFlow) for training.

```bash
cd datasets/waymo
uv sync          # one-time: build the isolated converter env (CPU torch + TensorFlow)
```

## Usage

Download the raw tfrecords first (Waymo v1.4.3, `individual_files/{training,validation}`
via `gsutil`), then run one conversion per split from this directory:

```bash
# validation
uv run python convert_waymo_to_coco.py \
  --tfrecord_dir /path/to/waymo_v1_4_3/validation \
  --work_dir     /path/to/waymococo_f0 \
  --image_dirname val2020 \
  --video_dirname validation_video_tensors \
  --image_filename_prefix val \
  --label_filename instances_val2020.json \
  --add_waymo_info --write_image

# training: swap validation -> training, val -> train,
#           validation_video_tensors -> training_video_tensors
```

Both `--video_dirname` and `--write_image` are required to emit the `.pt` tensors and JPEGs.

## Acknowledgements

The files under `waymo_open_dataset/` are vendored from the
[official Waymo Open Dataset code](https://github.com/waymo-research/waymo-open-dataset)
(Apache-2.0; see `waymo_open_dataset/LICENSE`) to avoid a heavy dependency. The converter
was informed by that code and by
[Waymo-Dataset-Tool](https://github.com/RalphMao/Waymo-Dataset-Tool).
