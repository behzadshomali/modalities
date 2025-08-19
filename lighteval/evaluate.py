import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main():
    # Tracker for saving results and optionally pushing to HF hub
    evaluation_tracker = EvaluationTracker(
        output_dir="/home/behzad_shomali/modalities/lighteval/results",
        save_details=True,
        push_to_hub=True,
        hub_results_org="Behzadshomali",
    )

    # Pipeline parameters: use Accelerate as the launcher
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        # override_batch_size=1,  # optional
        max_samples=10          # optional for quick debugging
    )

    # HuggingFace backend config instead of VLLM
    model_config = VLLMModelConfig(
    model_name="Behzadshomali/Multilingual5",
    trust_remote_code=True,
    # Optional: device_map="auto", torch_dtype="auto"
)

    task = "leaderboard|truthfulqa:mc|3|1"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    main()
