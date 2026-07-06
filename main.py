import argparse
import logging
import os
from datetime import timedelta

import torch
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from datasets import build_dataset
from models import build_model
from models.models import Stage2ModelProtocol
from trainer import build_trainer

# add the new imports for RAE training
from transport import Sampler, create_transport
from utils import (
    dump_config,
    get_experiments_dir,
    load_cfg,
    make_experiments_dir,
    setup_gpu_logger,
    to_dict,
)
from utils.collate import future_latents_collate
from utils.model_utils import instantiate_from_config


def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(
        backend="nccl", timeout=timedelta(minutes=600)
    )  # set timeout to 60 minutes for evaluation
    return local_rank, global_rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", required=True, help="Path to the cfg file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--profile", action="store_true", help="Enable torch profiler tracing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers per process",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., trainer.evaluation.eval_first=true)",
    )
    return parser.parse_args()


def main():
    local_rank, global_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    args = parse_args()
    experiment_dirs = get_experiments_dir(args.cfg_path, args.debug)
    if global_rank == 0:
        make_experiments_dir(
            experiment_dirs, overwrite=True
        )  # ensure main process owns directory creation
    barrier()
    gpu_log_path = experiment_dirs["log_dir"] / f"rank_{global_rank}.log"
    setup_gpu_logger(gpu_log_path, args.debug, global_rank)

    logger = logging.getLogger(__name__)
    logger.info(f"Rank {global_rank} initialized, logging at {gpu_log_path}.")

    # Log the experiment dir
    if global_rank == 0:
        experiment_dir = experiment_dirs["experiment_dir"]
        logger.info(f"Experiment directory: {experiment_dir}")
    cfg = load_cfg(args.cfg_path)

    # Apply command-line overrides
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    if args.debug:
        cfg.compile = False  # Otherwise can't debug the forward pass of the model which has been converted to a graph
        sampler_cfg = cfg.get("sampler", None)
        # if sampler_cfg is not None:
        #     sampler_cfg.params.num_steps = 2

    wandb_run = None
    if global_rank == 0:
        dump_config(cfg, experiment_dirs["resolved_config_path"])
        if not args.debug:
            wandb_run = wandb.init(
                project="FlowWM",
                name=os.path.splitext(os.path.basename(args.cfg_path))[0],
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    if wandb.run is not None:
        sweep_params = dict(wandb.config)

        def set_by_path(cfg_obj, key_path, value):
            parts = key_path.split(".")
            cur = cfg_obj
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = value

        for key, val in sweep_params.items():
            set_by_path(cfg, key, val)

    ####################################### let's mark the changes in the code for transport ##########################################
    # These configs are None for Heatmap and Deterministic Predictor, non None for Flow based models
    transport_cfg = to_dict(cfg.get("transport", None))
    misc = to_dict(cfg.get("misc", None))
    sampler_cfg = to_dict(cfg.get("sampler", None))

    if misc:
        time_dist_shift = float(misc.get("time_dist_shift", 1.0))

    transport = None
    if transport_cfg:
        transport_params = dict(
            transport_cfg.get("params", {})
        )  # let's check if this is correct
        transport_params.pop("time_dist_shift", None)

    eval_sampler_ode = None
    if sampler_cfg:
        sampler_mode = sampler_cfg.get("mode", "ODE").upper()
        sampler_params = dict(sampler_cfg.get("params", {}))
        transport = create_transport(
            **transport_params,
            time_dist_shift=time_dist_shift,
        )
        transport_sampler = Sampler(transport)
        if sampler_mode == "ODE":
            eval_sampler_ode = transport_sampler.sample_ode(**sampler_params)
        elif sampler_mode == "SDE":
            eval_sampler_ode = transport_sampler.sample_sde(**sampler_params)
        else:
            raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    ####################################### end of changes for transport ##########################################

    eval_only = cfg.eval_only
    eval_only_use_mini = cfg.eval_only_use_mini

    # assemble the dataset
    if eval_only:
        if eval_only_use_mini:
            eval_mini_dataset = build_dataset(cfg, split="val_mini")
            eval_dataset = None
        else:
            eval_dataset = build_dataset(cfg, split="val")
            eval_mini_dataset = None
        train_dataset = None
    else:
        eval_mini_dataset = build_dataset(cfg, split="val_mini")
        eval_dataset = build_dataset(cfg, split="val")
        train_dataset = build_dataset(
            cfg, split="train"
        )  # log on main rank if global_rank == 0: if train_dataset is not None: logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_mini_dataset is not None:
            logger.info(f"Eval mini dataset size: {len(eval_mini_dataset)}")
        if eval_dataset is not None:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=cfg.seed)
        if train_dataset is not None
        else None
    )
    eval_mini_sampler = (
        DistributedSampler(
            eval_mini_dataset, shuffle=False, seed=cfg.seed, drop_last=False
        )
        if eval_mini_dataset is not None
        else None
    )
    eval_sampler = (
        DistributedSampler(eval_dataset, shuffle=False, seed=cfg.seed, drop_last=False)
        if eval_dataset is not None
        else None
    )
    # assemble dataloaders from datasets
    persistent_workers = args.num_workers > 0

    train_loader = (
        DataLoader(
            dataset=train_dataset,
            batch_size=cfg.trainer.training.batch_size_per_gpu,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            sampler=train_sampler,
            collate_fn=future_latents_collate,
        )
        if train_dataset is not None
        else None
    )

    eval_mini_loader = (
        DataLoader(
            dataset=eval_mini_dataset,
            batch_size=cfg.trainer.evaluation.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            sampler=eval_mini_sampler,
            collate_fn=future_latents_collate,
        )
        if eval_mini_dataset is not None
        else None
    )

    eval_loader = (
        DataLoader(
            dataset=eval_dataset,
            batch_size=cfg.trainer.evaluation.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            sampler=eval_sampler,
            collate_fn=future_latents_collate,
        )
        if eval_dataset is not None
        else None
    )

    model_config = cfg.get("stage_2", None)
    if model_config is not None:
        model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    else:
        model = build_model(cfg)

    if "checkpoint_path" in cfg.trainer:
        checkpoint_path = cfg.trainer.checkpoint_path
        if global_rank == 0:
            logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        start_epoch = checkpoint.get("epoch", None)
        start_gradient_step = checkpoint.get("gradient_steps", None)
        if "ema_model_state_dict" in checkpoint:
            logger.info("Loading EMA weights from checkpoint")
            state_dict = checkpoint["ema_model_state_dict"]
        elif "model_state_dict" in checkpoint:
            logger.info("Loading model weights from checkpoint")
            state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Checkpoint does not contain model weights")

        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded checkpoint with strict=True (no rewriting).")
        except Exception as e:
            logger.warning(f"Direct load failed: {e}")

        cleaned = {}

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                cleaned[k[len("_orig_mod.") :]] = v
            else:
                cleaned[k] = v

        try:
            model.load_state_dict(cleaned, strict=True)
            logger.info(
                "successfully loaded checkpoint after stripping `_orig_mod.` prefix."
            )
        except Exception as e:
            logger.warning(f"Stripping `_orig_mod.` failed: {e}")

        if global_rank == 0:
            logger.info(
                f"Resuming from epoch {start_epoch}, gradient step {start_gradient_step}"
            )

    if global_rank == 0:
        model_param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model Parameters: {model_param_count / 1e6:.2f}M")

    model.to(device)

    compile_model = cfg.compile
    if global_rank == 0:
        logger.info("Compiling model? %s", compile_model)
    if compile_model:
        model = torch.compile(model)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=False,
        find_unused_parameters=True,
    )  # we set gradient_as_bucket_view to False to avoid potential issues with gradient clipping

    # Add weight decay for all parameters except positional encoding
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if "positional_encoding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": cfg.trainer.training.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=cfg.trainer.training.lr,
    )
    total_epochs = cfg.trainer.training.epochs
    warmup_epochs = cfg.trainer.training.warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=5e-2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )
    loss_cls = instantiate(cfg.loss)

    # Pass the full config to the evaluator without recursively instantiating every nested node
    evaluator_cfg = cfg.get("evaluator", None)
    evaluator = (
        instantiate(evaluator_cfg, cfg=cfg, _recursive_=False)
        if evaluator_cfg is not None
        else None
    )

    use_mixed_precision = cfg.trainer.get("use_mixed_precision", True)

    trainer = build_trainer(
        cfg=cfg,
        experiment_dirs=experiment_dirs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_mini_loader=eval_mini_loader,
        eval_loader=eval_loader,
        loss_cls=loss_cls,
        device=device,
        local_rank=local_rank,
        global_rank=global_rank,
        train_sampler=train_sampler,
        args=args,
        wandb_run=wandb_run,
        mixed_precision=use_mixed_precision,
        evaluator=evaluator,
        transport=transport,
        eval_sampler_ode=eval_sampler_ode,
    )

    if "checkpoint_path" in cfg.trainer:
        trainer.train_state.epoch = start_epoch
        trainer.train_state.gradient_steps = start_gradient_step

    # Only run profiler when explicitly requested
    if eval_only:
        trainer.run_evaluation(final_eval=True, eval_only_use_mini=eval_only_use_mini)
    elif args.debug and args.profile:
        trace_dir = experiment_dirs["trace_dir"]
        sort_key = "self_cuda_time_total"

        profiler_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        # Schedule: wait=0, warmup=5, active=2, repeat=1
        # This will warm up for 5 steps, then record 2 steps, then stop
        schedule = torch.profiler.schedule(
            wait=0,  # Don't wait, start warming up immediately
            warmup=5,  # Warm up for 5 gradient steps
            active=2,  # Record 2 gradient steps
            repeat=1,  # Do this cycle once then stop
        )

        def trace_handler(prof):
            """Save profiling traces when ready"""
            chrome_trace_path = trace_dir / "train_trace.json"
            summary_path = trace_dir / "train_trace.txt"

            # Export Chrome trace for visualization
            prof.export_chrome_trace(str(chrome_trace_path))
            print(f"Chrome trace saved to: {chrome_trace_path}")

            # Export summary table
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                # Limit the summary to the ten most time-consuming entries
                summary = prof.key_averages().table(sort_by=sort_key, row_limit=10)
                summary_file.write(summary)
            print(f"Profiler summary saved to: {summary_path}")

            # Optional: Also export memory timeline if CUDA is available
            if device.type == "cuda":
                memory_timeline_path = trace_dir / "memory_timeline.html"
                prof.export_memory_timeline(str(memory_timeline_path))
                print(f"Memory timeline saved to: {memory_timeline_path}")

            # Exit after saving traces in debug mode
            print("Profiling complete. Exiting due to debug mode.")
            import sys

            sys.exit(0)

        with profile(
            activities=profiler_activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=schedule,
            on_trace_ready=trace_handler,
        ) as prof:
            trainer.train(profiler=prof)
    else:
        # Normal training without profiler
        trainer.train(profiler=None)
    # log at the very end, Done!
    logger.info("Done!")
    destroy_process_group()


if __name__ == "__main__":
    # launch with torchrun --nnodes=1 --nproc_per_node=1 main.py --cfg_path configs/dinov3/heatmap/dinov3_train_latent_heatmap_predictor.yaml
    main()
