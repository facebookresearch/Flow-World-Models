from .trainer import TrainerHeatmap, TrainerDeterministic, TrainerFlowMatching
from .trainer_image_reconstruction import TrainerImageReconstruction

TRAINER_REGISTRY = {
    "trainer_heatmap": TrainerHeatmap,
    "trainer_deterministic": TrainerDeterministic,
    "trainer_flow_matching": TrainerFlowMatching,
    "trainer_image_reconstruction": TrainerImageReconstruction,
}


def build_trainer(
    cfg,
    experiment_dirs,
    model,
    optimizer,
    scheduler,
    train_loader,
    eval_mini_loader,
    eval_loader,
    loss_cls,
    device,
    local_rank,
    global_rank,
    train_sampler=None,
    wandb_run=None,
    mixed_precision=True,
    args=None,
    evaluator=None,
    transport=None,
    eval_sampler_ode=None,
):
    trainer_type = cfg.trainer.trainer_type
    if trainer_type not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported trainer '{trainer_type}'. Expected one of: {available}"
        )

    trainer_cls = TRAINER_REGISTRY[trainer_type]

    trainer_kwargs = dict(
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
        wandb_run=wandb_run,
        mixed_precision=mixed_precision,
        args=args,
        evaluator=evaluator,
        transport=transport,
        eval_sampler_ode=eval_sampler_ode,
    )

    trainer = trainer_cls(cfg, **trainer_kwargs)
    return trainer
