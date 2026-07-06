import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


def load_cfg(cfg_path, resolve=True):
    """Load a config file and resolve simple inheritance via an ``inherits`` key."""
    cfg_path = Path(cfg_path)
    cfg = OmegaConf.load(cfg_path)

    inherited_cfgs = []
    inherits = cfg.get("inherits")
    if inherits:
        for relative_path in inherits:
            parent_path = (cfg_path.parent / Path(relative_path)).resolve()
            inherited_cfgs.append(load_cfg(parent_path, resolve=False))
        cfg = OmegaConf.merge(*inherited_cfgs, cfg)
    if "inherits" in cfg:
        del cfg["inherits"]
    if resolve:
        OmegaConf.resolve(cfg)
    return cfg


def get_experiments_dir(cfg_path, debug):
    project_root = Path.cwd()
    cfg_path = Path(cfg_path).resolve()
    configs_root = project_root / "configs"
    experiment_root = project_root / "experiments"
    if debug:
        experiment_root /= "debug"
    relative_cfg_path = cfg_path.relative_to(configs_root)
    experiment_dir = (experiment_root / relative_cfg_path).with_suffix("")
    # now derive all sub directories from experiment dir.
    checkpoint_dir = experiment_dir / "checkpoints"
    dump_dir = experiment_dir / "dumps"
    log_dir = experiment_dir / "logs"
    resolved_config_dir = experiment_dir / "resolved_config"
    resolved_config_path = resolved_config_dir / f"resolve_{cfg_path.name}"
    log_path = log_dir / "train.log"
    trace_dir = experiment_dir / "trace"
    # return everything as a dictionnary. Every dir is a Path, for easier manipulatin later
    directories = {
        "experiment_dir": experiment_dir,
        "checkpoint_dir": checkpoint_dir,
        "dump_dir": dump_dir,
        "log_dir": log_dir,
        "resolved_config_dir": resolved_config_dir,
        "resolved_config_path": resolved_config_path,
        "log_path": log_path,
        "trace_dir": trace_dir,
    }
    return directories


def make_experiments_dir(dir_dict, overwrite):
    experiment_dir = Path(dir_dict["experiment_dir"])
    log_dir = Path(dir_dict.get("log_dir", experiment_dir / "logs"))
    if overwrite and experiment_dir.exists():
        for child in experiment_dir.iterdir():
            if child == log_dir:
                train_log = log_dir / "train.log"
                if train_log.exists():
                    train_log.unlink()
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    created_dirs = []
    for key, path in dir_dict.items():
        if key.endswith("_dir"):
            directory_path = Path(path)
            directory_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory_path)


def dump_config(cfg, resolved_config_path):
    resolved_cfg_path = Path(resolved_config_path)
    OmegaConf.save(cfg, resolved_cfg_path)
    logger = logging.getLogger(__name__)
    logger.info("Dumped resolved config to %s", resolved_cfg_path)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))


def setup_gpu_logger(log_path, debug, global_rank):
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.FileHandler(log_path)]
    if debug:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=level,
        format=f"[GPU {global_rank}] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def to_dict(cfg_section):
    if cfg_section is None:
        return {}
    return OmegaConf.to_container(cfg_section, resolve=True)
