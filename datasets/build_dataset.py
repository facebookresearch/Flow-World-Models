def build_dataset(config, split):
    dataset_name = config.datasets.dataset_name
    if dataset_name == "synthetic_bouncing_shapes":
        from .synthetic_bouncing_shapes import SyntheticBouncingShapesDataset

        return SyntheticBouncingShapesDataset(
            config.datasets[split].synthetic_bouncing_shapes
        )
    elif dataset_name == "waymo":
        from .waymo import WaymoVideoDataset

        return WaymoVideoDataset(config.datasets[split].waymo, split)
    elif dataset_name == "waymo_image":
        from .waymo_image import WaymoImageDataset

        return WaymoImageDataset(config.datasets[split].waymo_image, split)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
