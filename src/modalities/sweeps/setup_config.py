import hashlib
import logging
import logging.config
import os
import random
import re
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Type, TypeVar, Union

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator, model_validator
from rich.console import Console
from rich.logging import RichHandler

LOG_FILE_PATH: None | Path = None


class SlurmRun(BaseModel):
    job_id: str | None = os.environ.get("SLURM_JOB_ID", None)
    job_array_job_id: str | None = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    job_array_task_id: str | None = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    job_name: str | None = os.environ.get("SLURM_JOB_NAME", None)

    job_account: str | None = os.environ.get("SLURM_JOB_ACCOUNT", None)
    job_partition: str | None = os.environ.get("SLURM_JOB_PARTITION", None)
    job_qos: str | None = os.environ.get("SLURM_JOB_QOS", None)

    job_node_list: str | None = os.environ.get("SLURM_JOB_NODELIST", None)
    job_num_nodes: int | None = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

    job_start_time: str | None = os.environ.get("SLURM_JOB_START_TIME", None)
    job_time_limit: str | None = os.environ.get("SBATCH_TIMELIMIT", None)


class Experiment(BaseModel):
    experiment_name: str
    config_hash: str | None = None
    config_file_path: Path | None = None
    experiment_log_dir_path: Path = (
        Path("experiments") if not os.environ.get("WORKING_DIR") else Path(os.environ["WORKING_DIR"]) / "experiments"
    )
    log_file_path: Path | None = None
    start_time: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_level: int = logging.INFO
    seed: int = 42
    slurm_run: SlurmRun = SlurmRun()

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_logging_level(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            level_name = v.strip().upper()
            if not hasattr(logging, level_name):
                raise ValueError(f"Invalid logging level string: {v}")
            level = getattr(logging, level_name)
            if not isinstance(level, int):
                raise ValueError(f"Resolved logging level is not an int: {v}")
            return level
        raise TypeError(f"Expected int or str for logging level, got {type(v)}")

    @model_validator(mode="after")
    def set_experiment_log_dir_path(self) -> Self:
        # make this set idempotent, so that it can be called multiple times without changing the path
        if not str(self.experiment_log_dir_path).endswith(f"{self.experiment_name}/{self.start_time}"):
            new_path = self.experiment_log_dir_path / self.experiment_name / self.start_time
            new_path.mkdir(parents=True, exist_ok=True)
            self.experiment_log_dir_path = new_path
        return self


def store_config_file_with_hash_suffix(config_file_path: Path, dst_path: Path, uuid_str: str) -> None:
    out_config_file_path = dst_path / (config_file_path.stem + f"_{uuid_str}" + config_file_path.suffix)
    shutil.copyfile(config_file_path, out_config_file_path)


ExperimentConfig = TypeVar("ExperimentConfig", bound=Experiment)


def entry_point_setup(
    config_file_path: Path,
    class_type: Type[ExperimentConfig],
) -> ExperimentConfig:
    config: ExperimentConfig = load_config(
        config_file_path=config_file_path,
        class_type=class_type,
    )
    _set_config_hash(config, config_file_path)
    log_file_path = config.experiment_log_dir_path / f"log_{config.config_hash}_{config.start_time}.log"
    config.log_file_path = log_file_path
    setup_logging(
        log_file_path=log_file_path,
        log_level=config.log_level,
    )
    _setup_seeding(seed=config.seed)
    logging.info(config.model_dump_json(indent=4))
    return config


def _setup_seeding(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: int, log_file_path: Path | None = None, name: Optional[str] = None) -> None:
    """
    Configures logging to write to a file and to the console.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logging.info(f"Setting up logging with level: {log_level}, file: {log_file_path}, name: {name}")

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    if name:
        log_format = f"%(asctime)s - %(processName)s - {name} - %(levelname)s - %(name)s - %(message)s"
    else:
        log_format = "%(asctime)s - %(processName)s - %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    rich_handler = RichHandler(
        level=log_level,
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
        console=Console(),
    )
    logger.addHandler(rich_handler)


BaseModelConfig = TypeVar("BaseModelConfig", bound=BaseModel)


def load_config(config_file_path: Path, class_type: Type[BaseModelConfig]) -> BaseModelConfig:
    config_file_path = Path(config_file_path)
    conf_dict = load_omegaconf_config(config_file_path)
    config = class_type.model_validate(conf_dict)
    if isinstance(config, Experiment):
        config.config_file_path = config_file_path
    return config


def path_resolver(path: str | Path, command: str):
    path = Path(path)
    if command == "name":
        return path.name
    elif command == "parent":
        return path.parent
    elif command == "stem":
        return path.stem
    elif command == "suffix":
        return path.suffix


def load_omegaconf_config(config_path: Path | str, print_config: bool = False) -> dict:
    """Load a config file and resolve it with OmegaConf.
    Note: The fields "var_substitution" and "sweep" are not reserved, as they are used for sweep generations and easier variable
    substitutions with OmegaConf.
    """
    config_path = Path(config_path)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    if not OmegaConf.has_resolver("path"):
        OmegaConf.register_new_resolver("path", path_resolver)
    if not OmegaConf.has_resolver("var"):
        # pre-filled variables with error if not found
        experiments_dir = next((parent for parent in config_path.parents if parent.name == "experiments"), None)
        if experiments_dir is not None:
            try:
                # experiment runs
                exp_path_rel_exp_dir = config_path.parent.parent.parent.relative_to(experiments_dir)
                if str(exp_path_rel_exp_dir) == ".":
                    exp_path_rel_exp_dir = config_path.parent.relative_to(experiments_dir)
            except ValueError:
                # debug runs
                exp_path_rel_exp_dir = config_path.parent.relative_to(experiments_dir)
            exp_name = str(exp_path_rel_exp_dir).replace("/", "__")
            OmegaConf.register_new_resolver(
                "var",
                lambda key: dict(
                    exp_dir=exp_path_rel_exp_dir,
                    exp_name=exp_name,
                )[key],
            )
    conf = OmegaConf.load(config_path)

    if "sweep" in conf:
        # for sweep configs, we first want to create the sweep configs and resolve and interpolate afterwards
        sweep = conf.sweep
        conf.sweep = {}
        conf_dict = OmegaConf.to_container(conf, resolve=True)
        conf_dict = resolve_keys_in_container(conf_dict, OmegaConf.create(conf_dict))
        conf_dict["sweep"] = OmegaConf.to_container(sweep, resolve=False)
    else:
        conf_dict = OmegaConf.to_container(conf, resolve=True)
        conf_dict = resolve_keys_in_container(conf_dict, OmegaConf.create(conf_dict))

    if print_config:
        print(f"Resolved config:\n{OmegaConf.to_yaml(conf_dict)}")

    return conf_dict


JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def resolve_key_string(key_str: str, context: Any) -> str:
    """
    Resolve interpolations in a key string using OmegaConf.select.
    The context must be an OmegaConf container.
    """
    pattern = r"\$\{([^}]+)\}"

    def replacer(match):
        expr = match.group(1)  # e.g. "var_substitution.tool"
        try:
            value = OmegaConf.select(context, expr)
        except Exception as e:
            raise ValueError(f"Error resolving key expression '{expr}': {e}")
        return str(value)

    return re.sub(pattern, replacer, key_str)


def resolve_keys_in_container(node: JSONType, context: Any) -> JSONType:
    """
    Recursively process a container (dict or list) and resolve any interpolations in keys.
    """
    if isinstance(node, dict):
        new_dict = {}
        for key, value in node.items():
            new_key = resolve_key_string(key, context) if isinstance(key, str) else key
            new_dict[new_key] = resolve_keys_in_container(value, context)
        return new_dict
    elif isinstance(node, list):
        return [resolve_keys_in_container(item, context) for item in node]
    else:
        return node


def _set_config_hash(config: Experiment, config_path: Path) -> None:
    config.config_hash = _get_hash_sum_sha256_of_file(config_path)
    store_config_file_with_hash_suffix(config_path, config.experiment_log_dir_path, config.config_hash)


def _get_hash_sum_sha256_of_file(file_path: Path, last_n_chars: int = 7) -> str:
    hash = hashlib.sha256()
    bytes = bytearray(128 * 1024)
    mem_view = memoryview(bytes)
    with file_path.open("rb", buffering=0) as f:
        while n := f.readinto(mem_view):
            hash.update(mem_view[:n])
    return hash.hexdigest()[-last_n_chars:]


def pre_init(func):
    """Decorator to run a function before model initialization."""

    @model_validator(mode="before")
    @wraps(func)
    def wrapper(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Decorator to run a function before model initialization."""
        # Insert default values
        for name, field_info in cls.model_fields.items():
            if name not in values:
                if field_info.default_factory is not None:
                    values[name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[name] = field_info.default

        # Call the decorated function
        return func(cls, values)

    return wrapper
