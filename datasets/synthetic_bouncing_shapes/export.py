import shutil
import os
import gc
from pathlib import Path
from time import time

import torch
from itertools import cycle
from einops import rearrange
from tqdm import tqdm

from .generator import SyntheticBouncingShapesVideoGenerator
from utils import load_cfg
from utils.utils_visualisation import write_video


def reset_output_dirs(base_dir: Path):
    train_dir = base_dir / "train"
    eval_dir = base_dir / "evaluation"

    if train_dir.exists():
        print(f"Removing existing directory {train_dir}")
        shutil.rmtree(train_dir, ignore_errors=True)
    else:
        print(f"No existing directory at {train_dir}, nothing to erase.")

    if eval_dir.exists():
        print(f"Removing existing directory {eval_dir}")
        shutil.rmtree(eval_dir, ignore_errors=True)
    else:
        print(f"No existing directory at {eval_dir}, nothing to erase.")

    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return train_dir, eval_dir


def export_split(generator, output_dir):
    num_initial_conditions = (
        generator.get_num_initial_conditions()
    )  # This is the number of different combinations of velocity speed only
    target_total = generator.config.length
    progress_log_every = 1000
    trajectories_saved = 0
    initial_conditions_used = 0  # This is just a counter of how many ICs we have used to save trajectories but not unique (there can be repetitions). It serves as unique identifier to save trajectories
    set_of_used_ics = set()
    set_of_skipped_ics = set()
    print(
        f"num of initial conditions: {num_initial_conditions} for split {generator.config.split}"
    )
    print(
        f"target total trajectories to generate: {target_total} for split {generator.config.split}"
    )

    default_fps = generator.config.fps

    progress_kwargs = {
        "desc": f"Exporting {generator.config.split} dataset",
        "unit": "traj",
        "total": target_total,
        "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
    }
    progress = tqdm(**progress_kwargs)

    while trajectories_saved < target_total:
        for ic_idx in cycle(range(num_initial_conditions)):
            if trajectories_saved >= target_total:
                break

            num_trajectories = generator.get_num_trajectories(ic_idx)
            if num_trajectories == 1 or num_trajectories > 10:
                set_of_skipped_ics.add(ic_idx)
                print(f"Skipping IC {ic_idx} with {num_trajectories} trajectories")
                generator.release_initial_condition(ic_idx)
                continue
            else:
                set_of_used_ics.add(ic_idx)
                initial_conditions_used += 1

            ic_dir = output_dir / f"ic{initial_conditions_used:06d}"
            ic_dir.mkdir(parents=True, exist_ok=True)
            for traj_idx in range(num_trajectories):
                if trajectories_saved >= target_total:
                    break
                sample = generator.generate(ic_idx, traj_idx)
                video_tensor = rearrange(
                    sample["video"], "t c h w -> t h w c"
                ).contiguous()

                video_filename = (
                    ic_dir / f"ic{initial_conditions_used:06d}_traj{traj_idx:03d}.mp4"
                )
                target_filename = (
                    ic_dir / f"ic{initial_conditions_used:06d}_traj{traj_idx:03d}.pt"
                )
                # if one of these files already exist, raise an error
                if video_filename.exists() or target_filename.exists():
                    raise FileExistsError(
                        f"File {video_filename} or {target_filename} already exists."
                    )
                write_video(str(video_filename), video_tensor, fps=default_fps)
                torch.save(sample, target_filename)

                trajectories_saved += 1
                if (
                    trajectories_saved % progress_log_every == 0
                    or trajectories_saved == target_total
                ):
                    print(
                        f"[{generator.config.split}] saved {trajectories_saved}/{target_total} "
                        f"trajectories ({100.0 * trajectories_saved / target_total:.2f}%)"
                    )
                avg_traj_per_ic = trajectories_saved / initial_conditions_used
                progress.set_postfix(
                    {
                        "ic_progress": f"{ic_idx}/{num_initial_conditions}",
                        "avg_traj_per_ic": f"{avg_traj_per_ic:.2f}",
                    },
                    refresh=False,
                )
                progress.update(1)

            generator.release_initial_condition(ic_idx)
            gc.collect()

        progress.close()
        avg_trajectories = trajectories_saved / initial_conditions_used
        split_name = generator.config.split
        ics_exhausted = initial_conditions_used >= num_initial_conditions
        print(
            f"{split_name} split summary:\n"
            f"  trajectories generated: {trajectories_saved}\n"
            f"  unique initial conditions used: {len(set_of_used_ics)}/{num_initial_conditions}\n"
            f"  non unique initial conditions used: {initial_conditions_used}\n"
            f"  skipped initial conditions: {len(set_of_skipped_ics)}\n"
            f"  avg trajectories per IC: {avg_trajectories:.2f}\n"
            f"  exhausted all initial conditions: {'yes' if ics_exhausted else 'no'}"
        )


def main():
    start_time = time()
    base_path = Path(__file__).resolve()
    config_path = (
        base_path.parents[2]
        / "configs"
        / "datasets"
        / "synthetic_bouncing_shapes_export.yaml"
    )
    output_base = Path(
        os.environ.get(
            "SYNTHETIC_BOUNCING_SHAPES_ROOT",
            "/path/to/datasets/SyntheticBouncingShapes",
        )
    )

    cfg = load_cfg(config_path)
    train_cfg = cfg.generator.train
    eval_cfg = cfg.generator.evaluation

    train_dir, eval_dir = reset_output_dirs(output_base)

    eval_generator = SyntheticBouncingShapesVideoGenerator(eval_cfg)
    train_generator = SyntheticBouncingShapesVideoGenerator(train_cfg)

    export_split(eval_generator, eval_dir)
    export_split(train_generator, train_dir)
    elapsed_seconds = time() - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    print(f"Total export time: {hours}h {minutes:02d}m")


if __name__ == "__main__":
    main()
