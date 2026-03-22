from datetime import datetime
from enum import Enum
from typing import Callable, Optional

import torch

# try:
#     # Use try-except to avoid issues if _functorch is not available or config changed
#     import torch._functorch.config
#     # Fix for "compiled with non-empty donated buffers" error when using gradients with retain_graph=True
#     torch._functorch.config.donated_buffer = False
# except (ImportError, AttributeError):
#     pass

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from modalities.batch import DatasetBatch, EvaluationResultBatch, ResultItem
from modalities.checkpointing.stateful.app_state import AppState
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.messages import ExperimentStatus, MessageTypes, ProgressUpdate
from modalities.logging_broker.publisher import MessagePublisher
from modalities.loss_functions import Loss
from modalities.models.model import model_predict_batch
from modalities.models.parallelism.pipeline_parallelism import Pipeline
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_degree
from modalities.running_env.fsdp.reducer import Reducer
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF
from modalities.training.training_progress import TrainingProgress
from modalities.util import Aggregator, TimeRecorder, print_rank_0
from modalities.utils.mfu import MFUCalculatorABC


class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


class Trainer:
    def __init__(
        self,
        global_rank: int,
        progress_publisher: MessagePublisher[ProgressUpdate],
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        gradient_acc_steps: int,
        global_num_tokens_per_train_step: int,
        device_mesh: DeviceMesh | None,
        num_seen_train_steps: int,
        global_num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        gradient_clipper: GradientClipperIF,
        mfu_calculator: Optional[MFUCalculatorABC] = None,
    ) -> None:
        """
        Initializes the Trainer object.

        Args:
            global_rank (int): The global rank.
            progress_publisher (MessagePublisher[ProgressUpdate]): Progress publisher.
            evaluation_result_publisher (MessagePublisher[EvaluationResultBatch]): Evaluation result publisher.
            gradient_acc_steps (int): Gradient accumulation steps.
            global_num_tokens_per_train_step (int): Global number of tokens per train step.
            dp_degree (int): Data parallelism degree.
            pp_degree (int): Pipeline parallelism degree.
            num_seen_train_steps (int): Number of seen train steps.
            global_num_seen_tokens (int): Global number of seen tokens.
            num_target_steps (int): Number of target steps.
            num_target_tokens (int): Number of target tokens.
            gradient_clipper (GradientClipperIF): Gradient clipper.
            mfu_calculator (Optional[MFUCalculatorABC]): MFU calculator.

        Returns:
            None
        """
        self.global_rank = global_rank
        if device_mesh is not None:
            self.dp_degree = get_parallel_degree(
                device_mesh, [ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD]
            )
            self.pp_degree = get_parallel_degree(device_mesh, [ParallelismDegrees.PP])
        else:
            self.dp_degree = dist.get_world_size()
            self.pp_degree = 1
        self.progress_publisher = progress_publisher
        self.evaluation_result_publisher = evaluation_result_publisher
        self.gradient_acc_steps = gradient_acc_steps
        self.global_num_tokens_per_train_step = global_num_tokens_per_train_step
        self.num_seen_train_steps = num_seen_train_steps
        self.num_target_steps = num_target_steps
        self.num_target_tokens = num_target_tokens
        self.global_num_seen_tokens = global_num_seen_tokens
        self.gradient_clipper = gradient_clipper
        self.mfu_calculator = mfu_calculator

    @staticmethod
    def _get_num_train_steps_done(micro_batch_id: int, gradient_acc_steps: int) -> int:
        """
        Calculates the number of training steps done based on the micro batch ID and gradient accumulation steps.

        Args:
            micro_batch_id (int): The ID of the current micro batch.
            gradient_acc_steps (int): The number of gradient accumulation steps.

        Returns:
            int: The number of training steps done.
        """
        return (micro_batch_id + 1) // gradient_acc_steps

    def _perform_gradient_projection(
        self,
        model: FSDP,
        main_loss: torch.Tensor,
        aux_loss_list: list[torch.Tensor],
        scaling_factor: float,
        num_train_steps_done: int,
        loss_fun: Loss,
    ) -> None:
        """
        Performs PCGrad-style gradient projection for MTP losses.
        Projects auxiliary gradients onto the normal plane of the main task gradient if they conflict.
        """
        # Find parameters of GroupRecursiveGPT2MTPBlock
        target_params = []
        target_module_name = "GroupRecursiveGPT2MTPBlock"
        for module in model.modules():
            if module.__class__.__name__ == target_module_name:
                target_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # Remove duplicates
        target_params = list(set(target_params))
        
        # If no target params found (fallback to all params if MTP block missing but requested)
        if not target_params:
            target_params = [p for p in model.parameters() if p.requires_grad]

        # Optimization: Use flattened vectors to avoid python loop overhead
        def _get_flat_grads(model_params):
            views = []
            for p in model_params:
                if p.grad is not None:
                    g = p.grad
                    if hasattr(g, "to_local"):
                        g = g.to_local()
                    views.append(g.view(-1))
                else:
                    p_local = p
                    if hasattr(p, "to_local"):
                        p_local = p.to_local()
                    views.append(torch.zeros(p_local.numel(), device=p_local.device, dtype=p_local.dtype))
            return torch.cat(views)

        def _add_flat_grads_to_model(model_params, flat_grads):
            offset = 0
            for p in model_params:
                p_local = p
                if hasattr(p, "to_local"):
                    p_local = p.to_local()
                
                numel = p_local.numel()
                if numel > 0:
                    grad_slice = flat_grads[offset : offset + numel].view_as(p_local)
                    if p.grad is None:
                        if hasattr(p, "to_local"):
                             p.grad = torch.zeros_like(p)
                             p.grad.to_local().copy_(grad_slice)
                        else:
                             p.grad = grad_slice.clone()
                    else:
                        if hasattr(p.grad, "to_local"):
                             p.grad.to_local().add_(grad_slice)
                        else:
                             p.grad.add_(grad_slice)
                    offset += numel

        params = target_params

        # 1. Stash accumulated gradients from previous micro-batches (if any) for target params
        stashed_grads = {p: p.grad.clone() for p in params if p.grad is not None}
        
        # Clear grads for target params to separate computation
        # Note: Non-target params retain their accumulated gradients
        for p in params:
            p.grad = None
        
        # 2. Compute Main Task Gradient (Next Token Prediction)
        # retain_graph=True because we need to backward for aux losses later
        (main_loss / scaling_factor).backward(retain_graph=True)
        
        # Store main grads as flat vector just for target params
        main_grads_flat = _get_flat_grads(params)
        
        # Clear grads for target params again
        for p in params:
            p.grad = None
            
        # Initialize final_grads with main_grads
        final_grads_flat = main_grads_flat.clone()
        
        # Precompute main norm
        norm_main_sq = torch.dot(main_grads_flat, main_grads_flat)
        dist.all_reduce(norm_main_sq)
        norm_main = torch.sqrt(norm_main_sq)
        
        total_conflicts = 0
        cosine_sims = []
        
        monitor_conflicts = getattr(loss_fun, "monitor_gradient_conflicts", False)
        perform_projection = getattr(loss_fun, "perform_gradient_projection", False)

        # 3. Process each Auxiliary Task
        for i, aux_loss in enumerate(aux_loss_list):
            (aux_loss / scaling_factor).backward(retain_graph=True)
            
            aux_grads_flat = _get_flat_grads(params)
            
            # Consolidated All-Reduce
            dot_prod = torch.dot(main_grads_flat, aux_grads_flat)
            norm_aux_sq = torch.dot(aux_grads_flat, aux_grads_flat)
            
            stats = torch.stack([dot_prod, norm_aux_sq])
            dist.all_reduce(stats)
            dot_prod_global = stats[0]
            norm_aux_sq_global = stats[1]
            
            norm_aux = torch.sqrt(norm_aux_sq_global)
            
            if norm_main > 0 and norm_aux > 0:
                cos_sim = dot_prod_global / (norm_main * norm_aux)
            else:
                 cos_sim = torch.tensor(-10, device=main_grads_flat.device)

            cosine_sims.append(cos_sim.item())
            
            # Check for conflict
            if dot_prod_global < 0:
                total_conflicts += 1
                if perform_projection:
                    # Project aux gradient
                    # g_a_proj = g_a - (dot / |g_m|^2) * g_m
                    proj_coeff = dot_prod_global / (norm_main_sq + 1e-8)
                    
                    # final += g_a - proj * g_m
                    # We can do this in place on final_grads_flat
                    final_grads_flat.add_(aux_grads_flat).add_(main_grads_flat, alpha=-proj_coeff)
                else:
                    final_grads_flat.add_(aux_grads_flat)
            else:
                # No conflict, just add original aux gradient
                final_grads_flat.add_(aux_grads_flat)
            
            # Clear grads for target params for next aux task
            for p in params:
                 p.grad = None
            
        # 4. Restore stashed grads and add the computed final_grads
        for p, g in stashed_grads.items():
            p.grad = g # Restore accumulated gradients
            
        _add_flat_grads_to_model(params, final_grads_flat)
                
        # 5. Log stats to WandB if monitoring is enabled
        if monitor_conflicts:
            metrics = {
                "mtp_gradient_stats/num_conflicts": ResultItem(torch.tensor(float(total_conflicts)), decimal_places=2),
            }
            for i, sim in enumerate(cosine_sims):
                metrics[f"mtp_gradient_stats/cosine_sim_head_{i}"] = ResultItem(torch.tensor(sim), decimal_places=4)
                
            evaluation_result = EvaluationResultBatch(
                losses={},
                metrics=metrics,
                throughput_metrics={},
                dataloader_tag="mtp_gradient_projection",
                num_train_steps_done=num_train_steps_done,
            )
            self._publish_evaluation_result(
                evaluation_result_publisher=self.evaluation_result_publisher,
                evaluation_result=evaluation_result
            )

    def _train_batch(
        self,
        batch: DatasetBatch,
        model: FSDP,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fun: Loss,
        micro_batch_id: int,
        scheduled_pipeline: Optional[Pipeline] = None,
    ) -> tuple[bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Conducts a training step on batch of data.

        Args:
            batch (DatasetBatch): The input batch of data.
            model (FSDP): The model to train.
            optimizer (Optimizer): The optimizer used for training.
            scheduler (LRScheduler): The learning rate scheduler.
            loss_fun (Loss): The loss function used for training.
            micro_batch_id (int): The ID of the micro batch.
            scheduled_pipeline (Optional[Pipeline], optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            tuple[bool, int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
                A tuple containing the following:
                    - step_performed (bool): Indicates whether a training step was performed.
                    - num_train_steps_done (int): The number of training steps done.
                    - loss (Optional[torch.Tensor]): The computed loss.
                        None, if a non-last stage was processes in pipeline parallelism.
                    - gradient_norm_score (Optional[torch.Tensor]): The gradient norm score,
                        if a training step was performed otherwise return None.
        """
        if scheduled_pipeline is not None:
            pp_schedule = scheduled_pipeline.pp_schedule
            # Pipeline Parallel forward / backward inside step() call
            # with self.train_context(optional_context_parallel_ctx):
            targets, losses = (
                (batch.targets[loss_fun.target_key].contiguous(), [])
                if scheduled_pipeline.is_last_pp_stage
                else (None, None)
            )

            if scheduled_pipeline.is_first_pp_stage:
                pp_schedule.step(batch.samples[model.sample_key].contiguous(), target=targets, losses=losses)
            else:
                pp_schedule.step(target=targets, losses=losses)
            loss = torch.mean(torch.stack(losses)).to(losses[0].device) if scheduled_pipeline.is_last_pp_stage else None
        else:
            # else continue with loss calculation
            result_batch = model_predict_batch(model=model, batch=batch)
            loss = loss_fun(result_batch)
            
            aux_loss_list = None
            if isinstance(loss, tuple):
                if len(loss) == 4:
                    loss, ce_loss, aux_loss, aux_loss_list = loss
                elif len(loss) == 3:
                    loss, ce_loss, aux_loss = loss
                else:
                    loss = loss[0]
                    ce_loss = None
                    aux_loss = None
            else:
                ce_loss = None
                aux_loss = None
            
            # Check if gradient projection is requested or if monitoring is requested
            monitor_conflicts = getattr(loss_fun, "monitor_gradient_conflicts", False)
            perform_projection = getattr(loss_fun, "perform_gradient_projection", False)

            if aux_loss_list is not None and perform_projection:
                raise NotImplementedError("Gradient projection implementation should be reconsidered")
                # current_steps_done = Trainer._get_num_train_steps_done(
                #     micro_batch_id, self.gradient_acc_steps
                # )
                # self._perform_gradient_projection(
                #     model, 
                #     ce_loss, 
                #     aux_loss_list, 
                #     self.gradient_acc_steps,
                #     num_train_steps_done=current_steps_done,
                #     loss_fun=loss_fun
                # )
            else:
                (loss / self.gradient_acc_steps).backward()

        if (micro_batch_id + 1) % self.gradient_acc_steps == 0:
            gradient_norm_score = self.gradient_clipper.clip_gradients()
            optimizer.step()
            scheduler.step()
            # Step the mtp_lambda_scheduler if the loss function has one
            if hasattr(loss_fun, 'mtp_lambda_scheduler') and loss_fun.mtp_lambda_scheduler is not None:
                loss_fun.mtp_lambda_scheduler.step()
            if hasattr(loss_fun, 'efficiency_lambda_scheduler') and loss_fun.efficiency_lambda_scheduler is not None:
                loss_fun.efficiency_lambda_scheduler.step()
            optimizer.zero_grad()
            step_performed = True
        else:
            step_performed = False
            gradient_norm_score = None

        num_train_steps_done = Trainer._get_num_train_steps_done(
            micro_batch_id=micro_batch_id, gradient_acc_steps=self.gradient_acc_steps
        )

        self._track_recurrences_on_wandb(model, self.evaluation_result_publisher, num_train_steps_done)

        return step_performed, num_train_steps_done, loss, gradient_norm_score, ce_loss, aux_loss, aux_loss_list

    def train(
        self,
        app_state: AppState,
        train_loader: LLMDataLoader,
        loss_fun: Loss,
        training_log_interval_in_steps: int,
        evaluation_callback: Callable[[TrainingProgress], None],
        checkpointing_callback: Callable[[TrainingProgress], None],
        scheduled_pipeline: Pipeline | None = None,
    ):
        """
        Trains the model.

        Args:
            app_state (AppState): The application state containing the model, optimizer and lr scheduler.
            train_loader (LLMDataLoader): The data loader containing the training data.
            loss_fun (Loss): The loss function used for training.
            training_log_interval_in_steps (int): The interval at which training progress is logged.
            evaluation_callback (Callable[[TrainingProgress], None]): A callback function for evaluation.
            checkpointing_callback (Callable[[TrainingProgress], None]): A callback function for checkpointing.
            scheduled_pipeline (Pipeline | None, optional): In case of pipeline parallelism, this is used to
                operate the model. Defaults to None.

        Returns:
            None
        """
        model = app_state.model
        optimizer = app_state.optimizer
        lr_scheduler = app_state.lr_scheduler
        model.train()

        cumulated_losses = self._reset_tracked_losses()
        cumulated_losses_ce = self._reset_tracked_losses()
        cumulated_losses_aux = self._reset_tracked_losses()
        cumulated_losses_aux2 = self._reset_tracked_losses()

        # throughput
        thoughput_aggregator = Aggregator[ThroughputAggregationKeys]()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # batch loop
        batch: DatasetBatch
        # TODO: why do we need a barrier here?
        # dist.barrier()
        forward_backward_time_recorder = TimeRecorder()
        forward_backward_time_recorder.start()
        gradient_norm_scores = []

        # run evaluation callback and checkpointing callback before the first optimizer step
        evaluation_callback(num_train_steps_done=self.num_seen_train_steps)
        training_progress = TrainingProgress(
            num_seen_steps_previous_run=self.num_seen_train_steps,
            num_seen_tokens_previous_run=self.global_num_seen_tokens,
            num_seen_steps_current_run=0,
            num_seen_tokens_current_run=0,
            num_target_steps=self.num_target_steps,
            num_target_tokens=self.num_target_tokens,
        )
        checkpointing_callback(training_progress=training_progress)

        num_steps_todo = self.num_target_steps - self.num_seen_train_steps
        num_batches_todo = num_steps_todo * self.gradient_acc_steps
        # Because we might resume training, we add the starting batch id of the data loader
        for _, (micro_batch_id, batch) in zip(range(num_batches_todo), enumerate(train_loader)):
            # Train single batch
            (
                step_performed,
                num_train_steps_done,
                batch_loss,
                gradient_norm_score,
                ce_loss, 
                aux_loss,
                aux_loss2, # or ponder loss if using the temporal discounting ponder loss
            ) = self._train_batch(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                loss_fun=loss_fun,
                micro_batch_id=micro_batch_id,
                scheduled_pipeline=scheduled_pipeline,
            )
            forward_backward_time_recorder.stop()
            training_progress.num_seen_steps_current_run = num_train_steps_done
            training_progress.num_seen_tokens_current_run = self.global_num_tokens_per_train_step * num_train_steps_done

            # The batch_loss might be None if we use pipeline parallelism and are not the last stage.
            if batch_loss is not None:
                # Save the batch loss
                cumulated_losses[0] += batch_loss.item()
                # This works, because we always drop the last batch in case it has less samples than the batch size
                cumulated_losses[-1] += 1  # number of local batches

                if ce_loss is not None and aux_loss is not None:
                    cumulated_losses_ce[0] += ce_loss.item()
                    cumulated_losses_aux[0] += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                    cumulated_losses_aux2[0] += aux_loss2.item() if aux_loss2 is not None else -10.0
                    cumulated_losses_ce[-1] += 1
                    cumulated_losses_aux[-1] += 1
                    cumulated_losses_aux2[-1] += 1

            # gradient norm is already synced across all ranks
            if gradient_norm_score is not None:
                gradient_norm_scores.append(gradient_norm_score.item())

            batch_length_tensor = torch.tensor(len(batch)).to(device)
            thoughput_aggregator.add_value(key=ThroughputAggregationKeys.NUM_SAMPLES, value=batch_length_tensor)

            self._publish_progress(
                progress_publisher=self.progress_publisher,
                num_train_steps_done=training_progress.num_seen_steps_total,
                dataloader_tag=train_loader.dataloader_tag,
            )
            # Check if model performance should be logged
            if training_progress.num_seen_steps_total % training_log_interval_in_steps == 0 and step_performed:
                forward_backward_time = torch.tensor(forward_backward_time_recorder.delta_t).to(device)
                forward_backward_time_recorder.reset()

                thoughput_aggregator.add_value(
                    key=ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, value=forward_backward_time
                )
                # we only want to sync the num samples across data parallel ranks
                # so we divide the world size by the dp degree
                synced_num_samples = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.NUM_SAMPLES
                ) / (dist.get_world_size() / self.dp_degree)
                synced_forward_backward_time = thoughput_aggregator.get_all_reduced_value(
                    ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
                )
                synced_num_samples_per_second = synced_num_samples / synced_forward_backward_time
                # TODO: insert reducer from outside so Trainer is independent of FSDP
                # add the loss and gradient norm for the LAST batch

                cumulated_losses[1] = batch_loss.item() if batch_loss is not None else 0.0
                cumulated_losses_ce[1] = ce_loss.item() if ce_loss is not None else 0.0
                
                aux_loss_value = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                cumulated_losses_aux[1] = aux_loss_value if aux_loss is not None else 0.0
                cumulated_losses_aux2[1] = aux_loss2.item() if aux_loss2 is not None else -10.0

                reduced_losses = Reducer.reduce(
                    tensor=cumulated_losses,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * (world size / dp_degree))
                    # 2.) last batch loss / (world size / pp_degree)
                    post_processing_fun=lambda t: torch.stack(
                        [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                    ),
                )

                reduced_losses_ce = Reducer.reduce(
                    tensor=cumulated_losses_ce,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * (world size / dp_degree))
                    # 2.) last batch loss / (world size / pp_degree)
                    post_processing_fun=lambda t: torch.stack(
                        [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                    ),
                )
                reduced_losses_aux = Reducer.reduce(
                    tensor=cumulated_losses_aux,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * (world size / dp_degree))
                    # 2.) last batch loss / (world size / pp_degree)
                    post_processing_fun=lambda t: torch.stack(
                        [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                    ),
                )
                reduced_losses_aux2 = Reducer.reduce(
                    tensor=cumulated_losses_aux2,
                    operation=dist.ReduceOp.SUM,
                    # 1.) summed batch loss / (num batches * (world size / dp_degree))
                    # 2.) last batch loss / (world size / pp_degree)
                    post_processing_fun=lambda t: torch.stack(
                        [t[0] / t[-1], t[1] / dist.get_world_size() * self.pp_degree]
                    ),
                )
                losses = {
                    "train loss avg": ResultItem(reduced_losses[0], decimal_places=2),
                    "train loss last": ResultItem(reduced_losses[1], decimal_places=2),
                    "train ce loss avg": ResultItem(reduced_losses_ce[0], decimal_places=2),
                    "train ce loss last": ResultItem(reduced_losses_ce[1], decimal_places=2),
                    "train aux loss avg": ResultItem(reduced_losses_aux[0], decimal_places=2),
                    "train aux2 loss avg": ResultItem(reduced_losses_aux2[0], decimal_places=2),
                    "train aux loss last": ResultItem(reduced_losses_aux[1], decimal_places=2),
                    "train aux2 loss last": ResultItem(reduced_losses_aux2[1], decimal_places=2),
                }

                consumed_tokens = torch.tensor(training_progress.num_seen_tokens_total)
                metrics = {
                    "consumed tokens": ResultItem(consumed_tokens, 0),
                    "grad norm avg": ResultItem(torch.mean(torch.Tensor(gradient_norm_scores)), 2),
                    "grad norm last": ResultItem(torch.tensor(gradient_norm_scores[-1]), 2),
                }
                
                # Log mtp_lambda if a scheduler is being used
                if hasattr(loss_fun, 'mtp_lambda_scheduler') and loss_fun.mtp_lambda_scheduler is not None:
                    metrics["mtp_lambda"] = ResultItem(
                        torch.tensor(loss_fun.mtp_lambda_scheduler.mtp_lambda), decimal_places=4
                    )
                if hasattr(loss_fun, 'efficiency_lambda_scheduler') and loss_fun.efficiency_lambda_scheduler is not None:
                    metrics["efficiency_lambda"] = ResultItem(
                        torch.tensor(loss_fun.efficiency_lambda_scheduler.mtp_lambda), decimal_places=4
                    )
                
                gradient_norm_scores = []
                mfu_score = torch.tensor(-1.0)
                if self.mfu_calculator is not None:
                    mfu_score = self.mfu_calculator.compute(num_samples_per_second=synced_num_samples_per_second)

                # Collect peak memory depending on device type. On CPU we fall back to RSS (if available) or -1.
                if device.type == "cuda":
                    peak_memory_MB = torch.cuda.max_memory_allocated(device) / 1024**2  # in MB
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    # ru_maxrss is in kilobytes on Linux; convert to MB. Use -1.0 if resource unavailable.
                    try:
                        import resource  # Standard lib (POSIX). Not available on some platforms.

                        peak_memory_MB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    except Exception:
                        peak_memory_MB = -1.0

                training_metrics = EvaluationResultBatch(
                    losses=losses,
                    metrics=metrics,
                    # TODO: hardcoded metric key
                    throughput_metrics={
                        "train samples/s": ResultItem(synced_num_samples_per_second, 1),
                        "train mfu (16-bit)": ResultItem(mfu_score, 2),
                        "lr mean": ResultItem(torch.tensor(lr_scheduler.get_last_lr()).mean()),
                        "peak memory rank 0 (MB)": ResultItem(torch.tensor(peak_memory_MB), 2),
                    },
                    dataloader_tag=train_loader.dataloader_tag,
                    num_train_steps_done=training_progress.num_seen_steps_total,
                )
                print_rank_0(f"{datetime.now().isoformat(timespec='seconds')} | {training_metrics}")
                self._publish_evaluation_result(
                    evaluation_result_publisher=self.evaluation_result_publisher,
                    evaluation_result=training_metrics,
                )
                thoughput_aggregator.remove_keys()

                cumulated_losses_ce = self._reset_tracked_losses()
                cumulated_losses_aux = self._reset_tracked_losses()
                cumulated_losses_aux2 = self._reset_tracked_losses()
                cumulated_losses = self._reset_tracked_losses()
            if step_performed:
                evaluation_callback(num_train_steps_done=training_progress.num_seen_steps_total)
                checkpointing_callback(training_progress=training_progress)
            # we start the time recoder here again to also capture the time spend loading
            # via the dataloader.
            forward_backward_time_recorder.start()

    def _reset_tracked_losses(self):
        # Initializes and returns a tensor representing the cumulated loss and gradient norm.
        # The tensor is initialized with zeros and its device is set based on the availability of CUDA.

        cumulated_loss_and_gradient_norm = torch.zeros(3)
        if torch.cuda.is_available():
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to(torch.device("cuda"))
        else:
            cumulated_loss_and_gradient_norm = cumulated_loss_and_gradient_norm.to("cpu")
        return cumulated_loss_and_gradient_norm
    
    def _track_recurrences_on_wandb(self, model, evaluation_result_publisher, num_train_steps_done):
        if self.global_rank == 0:
            if hasattr(model, "recurrence_usage_stats"):
                recurrence_usage_stats = model.recurrence_usage_stats
                for layer_idx, recurrences in recurrence_usage_stats.items():
                    avg_recurrence = sum(recurrences) / len(recurrences)
                    metrics = EvaluationResultBatch(
                        losses={},
                        metrics={
                            f"recurrence_stats/avg_recurrence_layer_{layer_idx}": ResultItem(
                                torch.tensor(avg_recurrence), decimal_places=2
                            )
                        },
                        throughput_metrics={},
                        dataloader_tag="recurrence_stats",
                        num_train_steps_done=num_train_steps_done,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=evaluation_result_publisher,
                        evaluation_result=metrics,
                    )
        # Clear stats on ALL ranks to prevent memory leaks on non-zero ranks
        if hasattr(model, "recurrence_usage_stats"):
            model.recurrence_usage_stats = {}

        if self.global_rank == 0:
            if hasattr(model, "recurrence_embedding_cosine_similarity_stats"):
                recurrence_embedding_cosine_similarity_stats = model.recurrence_embedding_cosine_similarity_stats
                for layer_idx, similarities in recurrence_embedding_cosine_similarity_stats.items():
                    avg_similarity = sum(similarities) / len(similarities)
                    metrics = EvaluationResultBatch(
                        losses={},
                        metrics={
                            f"recurrence_embedding_stats/avg_cosine_similarity_layer_{layer_idx}": ResultItem(
                                torch.tensor(avg_similarity), decimal_places=4
                            ),                            
                        },
                        throughput_metrics={},
                        dataloader_tag="recurrence_embedding_stats",
                        num_train_steps_done=num_train_steps_done,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=evaluation_result_publisher,
                        evaluation_result=metrics,
                    )
        # Clear cosine similarity stats on ALL ranks
        if hasattr(model, "recurrence_embedding_cosine_similarity_stats"):
            model.recurrence_embedding_cosine_similarity_stats = {}

        if self.global_rank == 0:
            if hasattr(model, "recurrence_embedding_mse_similarity_stats"):
                recurrence_embedding_mse_similarity_stats = model.recurrence_embedding_mse_similarity_stats
                for layer_idx, similarities in recurrence_embedding_mse_similarity_stats.items():
                    avg_similarity = sum(similarities) / len(similarities)
                    metrics = EvaluationResultBatch(
                        losses={},
                        metrics={
                            f"recurrence_embedding_stats/avg_mse_similarity_layer_{layer_idx}": ResultItem(
                                torch.tensor(avg_similarity), decimal_places=4
                            ),                            
                        },
                        throughput_metrics={},
                        dataloader_tag="recurrence_embedding_stats",
                        num_train_steps_done=num_train_steps_done,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=evaluation_result_publisher,
                        evaluation_result=metrics,
                    )
        # Clear mse similarity stats on ALL ranks
        if hasattr(model, "recurrence_embedding_mse_similarity_stats"):
            model.recurrence_embedding_mse_similarity_stats = {}

        if self.global_rank == 0:
            if hasattr(model, "recurrence_logits_entropy_stats"):
                recurrence_logits_entropy_stats = model.recurrence_logits_entropy_stats
                for iter_idx, entropies in recurrence_logits_entropy_stats.items():
                    avg_entropy = sum(entropies) / len(entropies)
                    metrics = EvaluationResultBatch(
                        losses={},
                        metrics={
                            f"recurrence_entropy_stats/avg_logits_entropy_iter_{iter_idx}": ResultItem(
                                torch.tensor(avg_entropy), decimal_places=4
                            ),                            
                        },
                        throughput_metrics={},
                        dataloader_tag="recurrence_entropy_stats",
                        num_train_steps_done=num_train_steps_done,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=evaluation_result_publisher,
                        evaluation_result=metrics,
                    )
        # Clear entropy stats on ALL ranks
        if hasattr(model, "recurrence_logits_entropy_stats"):
            model.recurrence_logits_entropy_stats = {}

        if self.global_rank == 0:
            if hasattr(model, "halt_value_stats"):
                halt_value_stats = model.halt_value_stats
                for iter_idx, halt_vals in halt_value_stats.items():
                    avg_halt_signal =  torch.mean(halt_vals[0])
                    metrics = EvaluationResultBatch(
                        losses={},
                        metrics={
                            f"halt_stats/avg_halt_signal_iter_{iter_idx}": ResultItem(
                                avg_halt_signal.detach().clone() if isinstance(avg_halt_signal, torch.Tensor) else torch.tensor(avg_halt_signal),
                                decimal_places=4
                            ),                            
                        },
                        throughput_metrics={},
                        dataloader_tag="halt_stats",
                        num_train_steps_done=num_train_steps_done,
                    )
                    self._publish_evaluation_result(
                        evaluation_result_publisher=evaluation_result_publisher,
                        evaluation_result=metrics,
                    )
        # Clear halt stats on ALL ranks to prevent memory leaks
        if hasattr(model, "halt_value_stats"):
            model.halt_value_stats = {}

        if self.global_rank == 0:
            if hasattr(model, "gate_stats"):
                gate_stats = model.gate_stats
                for gate_key, layer_stats in gate_stats.items():
                    metrics_dict = {}
                    for layer_idx, iter_stats in layer_stats.items():
                         for iter_idx, gate_vals in iter_stats.items():
                             if len(gate_vals) > 0:
                                avg_gate_val = sum(gate_vals) / len(gate_vals)
                                metrics_dict[f"gate_stats/{gate_key}_layer_{layer_idx}_iter_{iter_idx}"] = ResultItem(
                                    torch.tensor(avg_gate_val), decimal_places=4
                                )
                    if metrics_dict:
                        metrics = EvaluationResultBatch(
                            losses={},
                            metrics=metrics_dict,
                            throughput_metrics={},
                            dataloader_tag="gate_stats",
                            num_train_steps_done=num_train_steps_done,
                        )
                        self._publish_evaluation_result(
                            evaluation_result_publisher=evaluation_result_publisher,
                            evaluation_result=metrics,
                        )
        # Clear gate stats on ALL ranks to prevent memory leaks
        if hasattr(model, "gate_stats"):
            model.gate_stats = {}
        
        if self.global_rank == 0:
            if hasattr(model, "gate_normalized_stats"):
                gate_normalized_stats = model.gate_normalized_stats
                for gate_key, layer_stats in gate_normalized_stats.items():
                    metrics_dict = {}
                    for layer_idx, iter_stats in layer_stats.items():
                         for iter_idx, gate_vals in iter_stats.items():
                             if len(gate_vals) > 0:
                                avg_gate_val = sum(gate_vals) / len(gate_vals)
                                metrics_dict[f"gate_stats/normalized_{gate_key}_layer_{layer_idx}_iter_{iter_idx}"] = ResultItem(
                                    torch.tensor(avg_gate_val), decimal_places=4
                                )
                    if metrics_dict:
                        metrics = EvaluationResultBatch(
                            losses={},
                            metrics=metrics_dict,
                            throughput_metrics={},
                            dataloader_tag="gate_normalized_stats",
                            num_train_steps_done=num_train_steps_done,
                        )
                        self._publish_evaluation_result(
                            evaluation_result_publisher=evaluation_result_publisher,
                            evaluation_result=metrics,
                        )
        # Clear gate normalized stats on ALL ranks to prevent memory leaks
        if hasattr(model, "gate_normalized_stats"):
            model.gate_normalized_stats = {}

        

    @staticmethod
    def _publish_progress(
        progress_publisher: MessagePublisher[ProgressUpdate],
        num_train_steps_done: int,
        dataloader_tag: str,
    ):
        # Publishes the progress of the training, i.e., number of training steps done.

        payload = ProgressUpdate(
            num_steps_done=num_train_steps_done,
            experiment_status=ExperimentStatus.TRAIN,
            dataloader_tag=dataloader_tag,
        )
        progress_publisher.publish_message(payload=payload, message_type=MessageTypes.BATCH_PROGRESS_UPDATE)

    @staticmethod
    def _publish_evaluation_result(
        evaluation_result_publisher: MessagePublisher[EvaluationResultBatch],
        evaluation_result: EvaluationResultBatch,
    ):
        # Publishes the evaluation result.

        evaluation_result_publisher.publish_message(
            payload=evaluation_result, message_type=MessageTypes.EVALUATION_RESULT
        )
