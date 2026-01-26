import glob
import json
import os
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, FilePath, field_validator, model_validator


class ClusterName(str, Enum):
    MN5 = "mn5"
    LEONARDO = "leonardo"
    CAPELLA = "capella"
    BOOSTER = "booster"
    DEVELBOOSTER = "develbooster"


def get_cuda_visible_devices() -> List[int]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return []
    return list(map(int, visible_devices.split(",")))


class SBatchArgs(BaseModel):
    job_name: str
    output: str
    error: str
    gpus_per_node: int
    nodes: int
    ntasks: Optional[int] = None
    partition: Optional[str] = None
    time: Optional[str] = None
    account: Optional[str] = None
    cpus_per_task: Optional[int] = None
    mem: int = 0  # in MB, 0 means no limit

    class Config:
        extra = "allow"

    def get_sbatch_directives(self) -> str:
        """Generates SBATCH directives dynamically."""
        directives = []
        for key, value in self.model_dump().items():
            if value is None:
                continue
            value = str(value)
            key = key.replace("_", "-")  # Convert snake_case to kebab-case for SBATCH directives
            directives.append(f"#SBATCH --{key}={value}")
        return "\n".join(directives)

    @field_validator("gpus_per_node")
    def validate_num_gpus(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError(
                f"Sweep generation logic does not support gpus_per_node={v}. "
                + "Always assume to use at least a single GPU, even if it is a CPU job."
            )
        return v


class ClusterConfig(BaseModel):
    # the defaults are specified in the config file pointed to by defaults_config_path
    name: ClusterName
    sbatch_args: SBatchArgs
    modules: List[str] = []
    defaults_config_path: FilePath | None = None

    @model_validator(mode="before")
    @classmethod
    def set_default_cluster_values(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "defaults_config_path" in data and data["defaults_config_path"]:
            with open(data["defaults_config_path"], "r") as f:
                cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
                clusters_defaults = ClusterDefaultsConfig(**cfg)

            for cluster in clusters_defaults.clusters:
                if cluster.name == data["name"]:
                    for key, default_value in cluster.model_dump().items():
                        if key not in data:
                            data[key] = default_value
                        else:
                            value = data.get(key)
                            if isinstance(value, dict):
                                default_value.update(value)
                                data[key] = default_value
                            if isinstance(value, list):
                                default_value = list(set(value) & set(default_value))
                                data[key] = default_value
                            elif value is None:  # we check not for empty as the value might be intentially empty
                                data[key] = default_value
        return data

    def get_cuda_visible_devices(self) -> List[int]:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return list(map(int, visible_devices.split(","))) if visible_devices else []

    def get_module_load_commands(self) -> str:
        """Formats and returns module loading commands dynamically."""
        if not self.modules:
            return ""
        return "\n".join([f"module load {module}" for module in self.modules])


class ClusterDefaultsConfig(BaseModel):
    clusters: List[ClusterConfig] = []


class SweepConfig(BaseModel):
    base_config_file_path: str | List[str] | None = None
    sweep: Dict[str, Any]
    cluster: ClusterConfig
    paired: List[List[str]] = []


class SBatchArrayJobGenerator:
    def __init__(
        self,
        *,
        sweep_config: SweepConfig,
        bash_script_file_path: FilePath,
        output_dir: Path,
        validate_config_class: Type[BaseModel] | None,
    ) -> None:
        """
        Initialize the SBatchArrayJobGenerator with cluster configuration, script configuration,
        sweep configuration, and script template.
        """
        self.sweep_config: SweepConfig = sweep_config
        self.bash_script_file_path = bash_script_file_path
        self.sweep_output_dir_path = output_dir
        self.sweep_output_dir_path.mkdir(exist_ok=True, parents=True)
        self.validate_config_class = validate_config_class

    def _load_base_configs(self) -> List[Dict[str, Any]]:
        if self.sweep_config.base_config_file_path is None:
            print("No base config file path provided. Using empty base config.")
            return [{}]
        patterns: List[str] = (
            self.sweep_config.base_config_file_path
            if isinstance(self.sweep_config.base_config_file_path, list)
            else [self.sweep_config.base_config_file_path]
        )
        base_configs: List[Dict[str, Any]] = []
        for pattern in patterns:
            matching_files = sorted(glob.glob(pattern, recursive=True))
            if not matching_files:
                raise ValueError(f"No files match the base config pattern: {pattern}")
            for file_path in matching_files:
                with open(file_path, "r") as f:
                    config = yaml.safe_load(f)
                base_configs.append(config)
        return base_configs

    def _generate_sweep_combinations(self) -> List[Dict[str, Any]]:
        return generate_nested_combinations(self.sweep_config.sweep)

    def _filter_combinations_by_pairs(
        self,
        combinations: List[Dict[str, Any]],
        paired: List[List[str]],
        sweep: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not paired:
            return combinations

        def extract_value(config, path):
            try:
                keys = path.split(".")
                value = config
                for key in keys:
                    value = value.get(key, None)
            except Exception as e:
                raise ValueError(
                    f"Error extracting value in config from path '{path}': {e}\n\n"
                    "Enable var_sibstitution when loading the config?"
                )
            return value

        def unzip(pairs: List[List[Any]]) -> List[List[Any]]:
            return list(zip(*pairs))

        valid_pairings = list(
            product(*[unzip([extract_value(sweep, path) for path in pair_paths]) for pair_paths in paired])
        )
        filtered_combinations = []
        selected_pairings = []
        for combo in combinations:
            pairing_in_combo = tuple([tuple(extract_value(combo, elem) for elem in pair) for pair in paired])
            if pairing_in_combo in valid_pairings:
                filtered_combinations.append(combo)
                selected_pairings.append(pairing_in_combo)
        assert set(_transform_to_tuple(selected_pairings)) == set(_transform_to_tuple(valid_pairings))
        return filtered_combinations

    def _remove_duplicate_dicts(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique_dicts = []
        for d in combinations:
            # Convert dict to a sorted JSON string
            serialized = json.dumps(d, sort_keys=True)
            if serialized not in seen:
                seen.add(serialized)
                unique_dicts.append(d)
        return unique_dicts

    def generate_sbatch_array_job_and_configs(self) -> None:
        base_configs = self._load_base_configs()
        sweep_combinations = self._generate_sweep_combinations()
        if not sweep_combinations:
            raise ValueError(
                "No sweep combinations generated. Check if there are empty arrays in the sweep config "
                "and turn them into lists: [] -> [[]]"
            )
        sweep_combinations = self._filter_combinations_by_pairs(
            sweep_combinations, self.sweep_config.paired, sweep=self.sweep_config.sweep
        )
        sweep_combinations = self._remove_duplicate_dicts(sweep_combinations)
        configs_dir = self.sweep_output_dir_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        config_paths = []
        job_index = 0
        for base_config in base_configs:
            for sweep_params in sweep_combinations:
                merged_config = deep_merge(deepcopy(base_config), sweep_params)
                # after merging, we resolve the sweep-part of the config
                merged_config = OmegaConf.to_container(OmegaConf.create(merged_config), resolve=False)
                if "experiment_name" in merged_config:
                    merged_config["experiment_name"] += f"_job_{job_index}"
                config_path = configs_dir / f"config_{job_index}.yaml"
                if self.validate_config_class is not None:
                    # we only resolve to check the config validity, the real resolving should be done on the client
                    self.validate_config_class(
                        **dict(OmegaConf.to_container(OmegaConf.create(merged_config), resolve=True))
                    )
                with config_path.open("w") as f:
                    yaml.safe_dump(merged_config, f, default_flow_style=False, sort_keys=False)
                print(f"Configuration file generated: {config_path}")
                config_paths.append(str(config_path))
                job_index += 1
        self.print_expansion_keys()
        sbatch_file_path = self._write_sbatch_file(config_paths=config_paths)
        self._write_screen_job_bash_file(
            sbatch_file_path=sbatch_file_path,
            config_paths=config_paths,
        )

    def _write_sbatch_file(
        self,
        *,
        config_paths: List[str],
    ) -> Path:
        # Validate and make sure the output and error log paths are absolute
        if not Path(self.sweep_config.cluster.sbatch_args.output).is_absolute():
            self.sweep_config.cluster.sbatch_args.output = str(
                self.sweep_output_dir_path / self.sweep_config.cluster.sbatch_args.output
            )
        if not Path(self.sweep_config.cluster.sbatch_args.error).is_absolute():
            self.sweep_config.cluster.sbatch_args.error = str(
                self.sweep_output_dir_path / self.sweep_config.cluster.sbatch_args.error
            )
        for log_path in [
            self.sweep_config.cluster.sbatch_args.output,
            self.sweep_config.cluster.sbatch_args.error,
        ]:
            if any([slurm_placeholder in str(Path(log_path).parent) for slurm_placeholder in ["%x", "%A", "%a", "%j"]]):
                raise ValueError(
                    f"Do not use slurm env vars in log paths except the file name: {Path(log_path).parent}"
                )
            Path(log_path).parent.mkdir(exist_ok=True)

        # Write the file
        sbatch_array_size = len(config_paths)
        sbatch_file_path = self.sweep_output_dir_path / f"run_slurm_{self.sweep_config.cluster.name.value}.sbatch"
        # We need to use relative paths, to make the script work across different machines with different .env files
        working_dir = next(
            (parent for parent in self.bash_script_file_path.parents if parent.name == "experiments"), None
        ).parent
        with sbatch_file_path.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write(self.sweep_config.cluster.sbatch_args.get_sbatch_directives() + "\n")
            # We make sure the slurm working dir is
            f.write(f"#SBATCH --chdir={working_dir}\n")
            f.write(f"#SBATCH --array=0-{sbatch_array_size - 1}\n\n")
            f.write(self.sweep_config.cluster.get_module_load_commands() + "\n\n")

            f.write("set -ex\n\n")

            f.write("# if WORKING_DIR is not set, use the current directory\n")
            f.write('if [ -z "$WORKING_DIR" ]; then\n')
            f.write("  WORKING_DIR=$(pwd)\n")
            f.write("fi\n")

            # This will load VENV_PATH, WORKING_DIR, and other environment variables from the .env file
            # We don't need it here, but maybe user-defined other variables in the .env file
            f.write(
                f'export EXPERIMENT_DIR="$WORKING_DIR/{self.bash_script_file_path.parent.relative_to(working_dir)}"\n'
            )
            f.write("export $(grep -v '^#' $EXPERIMENT_DIR/.env | xargs)\n")
            f.write("cd $WORKING_DIR\n")
            f.write("conda activate /projects/p_gptx/behzad_shomali/conda_envs/modalities_sh_cloned\n")

            f.write("CONFIG_FILES=(\n")
            for path in config_paths:
                f.write(f'  "{Path(path).relative_to(working_dir)}"\n')
            f.write(")\n\n")
            f.write("export CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}\n\n")
            f.write(f"source {self.bash_script_file_path.relative_to(working_dir)}\n")

        print(f"SBATCH script generated: {sbatch_file_path}")
        return sbatch_file_path

    def _write_screen_job_bash_file(
        self,
        *,
        sbatch_file_path: Path,
        config_paths: List[str],
    ) -> None:
        screen_jobs_bash_file_path = self.sweep_output_dir_path / "run_local.sh"
        job_name = self.sweep_config.cluster.sbatch_args.job_name
        num_configs = len(config_paths)
        screen_logs_dir_path = sbatch_file_path.parent / "screen_logs"
        screen_logs_dir_path.mkdir(exist_ok=True)
        gpus_available = self.sweep_config.cluster.get_cuda_visible_devices()
        if not gpus_available:
            print("Error! No GPUs available. Generating screen job file with a single GPU (0).")
            gpus_available = [0]
        gpus_per_node = self.sweep_config.cluster.sbatch_args.gpus_per_node
        if gpus_per_node > len(gpus_available):
            print(
                "Error! Number of GPUs per job is greater than the number of available GPUs."
                + f" Creating example screen job file with available GPUs {gpus_available}."
            )
            gpus_per_node = 1
        num_groups = len(gpus_available) // gpus_per_node
        if num_groups == 0:
            num_groups = 1
        tasks_per_group = num_configs // num_groups
        remainder = num_configs % num_groups
        groups = {}
        start_idx = 0
        for group in range(num_groups):
            extra = 1 if group < remainder else 0
            end_idx = start_idx + tasks_per_group + extra - 1
            assigned_gpus = gpus_available[group * gpus_per_node : group * gpus_per_node + gpus_per_node]
            cuda_devices = ",".join(map(str, assigned_gpus))
            gpu_ids = cuda_devices.replace(",", "_")
            tasks = list(range(start_idx, end_idx + 1))
            if tasks:
                groups[gpu_ids] = {"cuda_devices": cuda_devices, "tasks": tasks}
                start_idx = end_idx + 1
        with screen_jobs_bash_file_path.open("w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"SBATCH_FILE={sbatch_file_path}\n\n")
            for gpu_ids, data in groups.items():
                tasks = data["tasks"]
                if len(tasks) == 0:
                    raise ValueError(
                        "No tasks assigned to this group. "
                        + f"Check the number of available GPUs ({gpus_available}) and "
                        + f"number of GPUs needed per task ({self.sweep_config.cluster.sbatch_args.gpus_per_node}) and "
                        + f"number of tasks ({num_configs})."
                    )
                loop_range = f"{{{tasks[0]}..{tasks[-1]}}}"
                cuda_devices = data["cuda_devices"]
                # screen -dmS eval_lm_arena__gen_answer_dgx bash -c "for i in {0..0}; do SLURM_ARRAY_TASK_ID=\$i CUDA_VISIBLE_DEVICES='4,5,6,7' bash $SBATCH_FILE > /raid/s3/opengptx/alexw/pipeline/experiments/eval_lm_arena/gen_answer_dgx/jobs/screen_logs/local_job_\$i_screen.log 2>&1; done"
                f.write(
                    f"screen -dmS {job_name.replace('/', '_')}_gpus_{gpu_ids} bash -c "
                    + f'"for i in {loop_range}; do '
                    + f"SLURM_ARRAY_TASK_ID=\\$i CUDA_VISIBLE_DEVICES='{cuda_devices}' bash $SBATCH_FILE "
                    + f'> "{screen_logs_dir_path}/screen_job_\\${{i}}_gpu_{gpu_ids}_screen.log" 2>&1; '
                    + 'done"\n\n'
                )
        print(f"Screen job script generated: {screen_jobs_bash_file_path}")

    def print_expansion_keys(self) -> None:
        def find_expansion_keys(sweep_dict: Dict[str, Any], current_path: str = "") -> List[str]:
            """
            Recursively finds key paths in a dictionary that lead to a list with more than one element.
            These are the keys that will cause an expansion in the number of jobs.
            """
            expansion_keys = []
            for key, value in sweep_dict.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, list) and len(value) > 1:
                    expansion_keys.append(new_path)
                elif isinstance(value, dict):
                    expansion_keys.extend(find_expansion_keys(value, new_path))
            return expansion_keys

        expansion_keys = find_expansion_keys(self.sweep_config.sweep)
        paired_expansion_keys = defaultdict(list)
        # group keys that are paired into arrays
        for idx, key in enumerate(expansion_keys):
            for pair in self.sweep_config.paired:
                if key in pair:
                    paired_expansion_keys[tuple(pair)].append(key)
                else:
                    paired_expansion_keys[(key,)].append(key)

        if paired_expansion_keys:
            print("\nThe following key paths caused config expansion:")
            for key_paths in paired_expansion_keys.values():
                print(f"- {key_paths}")
        else:
            print("\nNo expansion keys found. A single config was generated.")


def generate_nested_combinations(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    def expand(sweep_dict):
        if isinstance(sweep_dict, dict):
            if not sweep_dict:
                return [{}]
            keys, values = zip(*((k, expand(v)) for k, v in sweep_dict.items()))
            return [dict(zip(keys, combo)) for combo in product(*values)]
        elif isinstance(sweep_dict, list):
            return sweep_dict
        else:
            return [sweep_dict]

    return expand(sweep)


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _transform_to_tuple(inp: Any) -> Any:
    if isinstance(inp, tuple):
        return tuple(_transform_to_tuple(i) for i in inp)
    if isinstance(inp, list):
        return tuple(_transform_to_tuple(i) for i in inp)
    elif isinstance(inp, dict):
        return tuple({k: _transform_to_tuple(v) for k, v in inp.items()}.items())
    else:
        return inp
