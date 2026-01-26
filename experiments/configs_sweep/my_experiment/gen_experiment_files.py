from pathlib import Path
import sys



from modalities.sweeps.sbatch_array_job_generator import SBatchArrayJobGenerator, SweepConfig
from modalities.sweeps.setup_config import load_config
# from datadistill.configs.pipeline_config import PipelineConfig
# from datadistill.configs.setup_config import load_config
# from datadistill.experiments.sbatch_array_job_generator import SBatchArrayJobGenerator, SweepConfig

if __name__ == "__main__":
    experiment_path = Path(__file__).parent
    pattern = "sweep_*config.yaml"
    sweep_config_paths = list(experiment_path.glob(pattern))
    if not sweep_config_paths:
        raise ValueError(f"No {pattern} files found in {experiment_path}")
    if len(sweep_config_paths) > 1:
        raise ValueError(f"Multiple {pattern} files found in {experiment_path}")

    sweep_config_path = sweep_config_paths[0]
    
    sweep_config: SweepConfig = load_config(sweep_config_path, SweepConfig)
    generator = SBatchArrayJobGenerator(
        sweep_config=sweep_config,
        bash_script_file_path=experiment_path / "script.sh",
        output_dir=sweep_config_path.parent / "jobs",
        validate_config_class=None,
    )
    generator.generate_sbatch_array_job_and_configs()
