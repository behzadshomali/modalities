
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
import sys
from pathlib import Path
import wandb


from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent)) 

from pondering_lm.custom_modules import PonderingModelConfig, PonderingModelForCausalLM
from modalities.__main__ import Main
from modalities.batch import DatasetBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv




def main():
    # load and parse the config file
    # cwd = Path(__file__).parent
    # # change to cwd
    # os.chdir(cwd)
    # config_file_path = cwd / Path("config_lorem_ipsum.yaml")
    config_file_path = Path("/raid/s3/opengptx/behzad_shomali/modalities/config_files/training/fineweb2_edu_pondering.yaml")

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        # instantiate the Main entrypoint of modalities by passing in the config path
        modalities_main = Main(config_path=config_file_path)

        # add the custom component to modalities
        modalities_main.add_custom_component(
            component_key="model",
            variant_key="custom_pondering_llama",
            custom_component=PonderingModelForCausalLM,
            custom_config=PonderingModelConfig,
        )
        # run the experiment
        components: TrainingComponentsInstantiationModel = modalities_main.build_components(
            components_model_type=TrainingComponentsInstantiationModel
        )
        modalities_main.run(components)

        # if not isinstance(components.__dict__["app_state"]._model._fsdp_wrapped_module, PonderingModelForCausalLM):
        #     raise ValueError("Custom model was not used.")
        # else: 
        #     print("Custom model was successfully used.")


if __name__ == "__main__":
    main()
    wandb.finish()