from abc import ABC, abstractmethod
from typing import List, Tuple, overload

import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch


class Loss(ABC):
    def __init__(self, tag: str):
        self._tag = tag

    @property
    def tag(self) -> str:
        return self._tag

    @abstractmethod
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Calculates the loss
        :return: Loss tensor
        """
        raise NotImplementedError


class CLMCrossEntropyLoss(Loss):
    def __init__(self, target_key: str, prediction_key: str, tag: str = "CLMCrossEntropyLoss"):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        # Mean over the tokens in the local-batch (batch per rank)
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, lm_logits = self._parse_arguments(args, kwargs)

        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()
        # Flatten the tokens. We compute here, the loss per token.
        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def _parse_arguments(
        self,
        args: list[torch.Tensor] | list[InferenceResultBatch],
        kwargs: dict[str, torch.Tensor] | dict[str, InferenceResultBatch],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        return labels, lm_logits
    

class CLMRecurrenceCosineSimPenaltyLoss(CLMCrossEntropyLoss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        penalty_alpha: float = 1.0, 
        tag: str = "CLMRecurrenceCosineSimPenaltyLoss"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve predictions from the batch.
            penalty_alpha: Coefficient (lambda) for the recurrence penalty term.
            tag: Name of the loss module.
        """
        super().__init__(target_key, prediction_key, tag)
        self.penalty_alpha = penalty_alpha

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # 1. Parse arguments using the parent class logic to get raw inputs
        labels, raw_outputs = self._parse_arguments(args, kwargs)

        # 2. Extract Logits and Penalty
        # Check if the output is the dictionary format provided in the prompt
        recurrence_embedding_cosine_similarity = torch.tensor(-10.0, device=labels.device)
        recurrence_embedding_mse_similarity = torch.tensor(-10.0, device=labels.device)
        
        if isinstance(raw_outputs, dict):
            # Extract logits (h)
            lm_logits = raw_outputs.get("logits")
            
            # Extract the similarity penalty if it exists
            if "recurrence_embedding_cosine_similarity" in raw_outputs:
                recurrence_embedding_cosine_similarity = raw_outputs["recurrence_embedding_cosine_similarity"]
                
                # Ensure penalty is on the correct device
                if recurrence_embedding_cosine_similarity.device != lm_logits.device:
                    recurrence_embedding_cosine_similarity = recurrence_embedding_cosine_similarity.to(lm_logits.device)
            if "recurrence_embedding_mse_similarity" in raw_outputs:
                recurrence_embedding_mse_similarity = raw_outputs["recurrence_embedding_mse_similarity"]
                
                # Ensure penalty is on the correct device
                if recurrence_embedding_mse_similarity.device != lm_logits.device:
                    recurrence_embedding_mse_similarity = recurrence_embedding_mse_similarity.to(lm_logits.device)
        else:
            # Fallback: Assumption that raw_outputs are just the logits (standard behavior)
            lm_logits = raw_outputs

        # 3. Standard Cross Entropy Calculation
        # Move labels to correct device
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()

        # Flatten tokens for CE Loss
        ce_loss = self.loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # 4. Combine Losses
        # L_total = L_ce + (weight * L_penalty)

        # (1,1), (0,1), (0.5,0) --> y = 4x^2 - 4x + 1
        penalty = 4 * (recurrence_embedding_cosine_similarity ** 2) - 4 * recurrence_embedding_cosine_similarity + 1
        total_loss = ce_loss + (self.penalty_alpha * penalty)

        return total_loss, ce_loss, self.penalty_alpha * penalty
    

class CLMRecurrenceEntropyPenaltyLoss(CLMCrossEntropyLoss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        penalty_alpha: float = 1.0, 
        tag: str = "CLMRecurrenceEntropyPenaltyLoss"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve predictions from the batch.
            penalty_alpha: Coefficient (lambda) for the recurrence penalty term.
            tag: Name of the loss module.
        """
        super().__init__(target_key, prediction_key, tag)
        self.penalty_alpha = penalty_alpha

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # 1. Parse arguments using the parent class logic to get raw inputs
        labels, raw_outputs = self._parse_arguments(args, kwargs)

        
        
        if isinstance(raw_outputs, dict):
            # Extract logits (h)
            lm_logits = raw_outputs.get("logits")
            
            # Extract the each_recurrence_entropy if it exists
            if "each_recurrence_entropy" in raw_outputs:
                each_recurrence_entropy = raw_outputs["each_recurrence_entropy"]
                
                # Ensure penalty is on the correct device
                if each_recurrence_entropy.device != lm_logits.device:
                    each_recurrence_entropy = each_recurrence_entropy.to(lm_logits.device)

        else:
            raise ValueError("Expected raw_outputs to be a dictionary containing at least 'logits' and 'each_recurrence_entropy'.")

        # 3. Standard Cross Entropy Calculation
        # Move labels to correct device
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()

        # Flatten tokens for CE Loss
        ce_loss = self.loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        entropy_weights = torch.tensor([i for i in range(1, len(each_recurrence_entropy)+1)], device=lm_logits.device, dtype=each_recurrence_entropy.dtype) # TODO: maybe start from 0 so we don't care about the first iteration entropy as it always exists
        normalized_entropy_weights = entropy_weights / entropy_weights.sum()
        penalty = sum(e * w for e, w in zip(each_recurrence_entropy, normalized_entropy_weights))

        total_loss = ce_loss + (self.penalty_alpha * penalty)
        return total_loss, ce_loss, self.penalty_alpha * penalty


class CLMRecurrenceWeightedLoss(CLMCrossEntropyLoss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        penalty_alpha: float = 1.0, 
        tag: str = "CLMRecurrenceWeightedLoss"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve predictions from the batch.
            penalty_alpha: Coefficient (lambda) for the recurrence penalty term.
            tag: Name of the loss module.
        """
        super().__init__(target_key, prediction_key, tag)
        self.penalty_alpha = penalty_alpha

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        # 1. Parse arguments using the parent class logic to get raw inputs
        labels, raw_outputs = self._parse_arguments(args, kwargs)

        
        
        if isinstance(raw_outputs, dict):
            # Extract logits (h)
            # lm_logits = raw_outputs.get("logits")
            
            # Extract the each_recurrence_logits if it exists
            # if "each_recurrence_logits" in raw_outputs:
            each_recurrence_logits = raw_outputs["each_recurrence_logits"]
            device = each_recurrence_logits[0].device
            
            # Ensure penalty is on the correct device
            # if each_recurrence_logits.device != lm_logits.device:
            #     each_recurrence_logits = each_recurrence_logits.to(lm_logits.device)
        else:
            raise ValueError("Expected raw_outputs to be a dictionary containing 'each_recurrence_logits'.")

        # 3. Standard Cross Entropy Calculation
        # Move labels to correct device
        labels = labels.to(device)
        shift_labels = labels.contiguous().long()

        ce_loss_list = []
        for r in range(len(each_recurrence_logits)):
            lm_logits = each_recurrence_logits[r]
            shift_logits = lm_logits.contiguous()

            # Flatten tokens for CE Loss
            ce_loss = self.loss_fun(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            ce_loss_list.append(ce_loss)

        loss_weights = torch.tensor([i for i in range(1, len(ce_loss_list)+1)], device=device, dtype=each_recurrence_logits[0].dtype)
        normalized_loss_weights = loss_weights / loss_weights.sum()

        total_loss = sum(l * w for l, w in zip(ce_loss_list, normalized_loss_weights))
        return total_loss, ce_loss_list[-1], torch.tensor(0.0)  # return the last ce_loss as representative


class MTPCrossEntropyLoss(Loss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        mtp_prediction_key: str = "mtp_logits", 
        mtp_lambda: float = 1.0,
        perform_gradient_projection: bool = False,
        tag: str = "MTPCrossEntropyLoss"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve the MAIN head logits.
            mtp_prediction_key: Key to retrieve the list of MTP head logits.
            mtp_lambda: Weighting factor for the MTP auxiliary loss. 
                        Paper suggests 1.0 or similar.
            perform_gradient_projection: If True, returns detailed losses to allow gradient projection.
            tag: Tag for logging.
        """
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.mtp_prediction_key = mtp_prediction_key
        self.mtp_lambda = mtp_lambda
        self.perform_gradient_projection = perform_gradient_projection
        
        # Mean over the tokens in the local-batch
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, mtp_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, _, mtp_logits_list = self._parse_arguments(args, kwargs)

        # Move labels to correct device
        labels = labels.to(mtp_logits_list[0].device).long()
        shift_logits = mtp_logits_list[0].contiguous()
        shift_labels = labels.contiguous().long()
        
        # --- 1. Main Head Loss (Next Token Prediction) ---
        # If input is x_1...x_t, labels should be x_2...x_{t+1}
        
        # regular next-token prediction loss
        # based on the first iteration logits
        next_token_loss = self.loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # --- 2. MTP Heads Loss ---
        # if mtp_logits_list:
        mtp_loss_sum = 0.0
        mtp_component_losses = []
        
        for i, mtp_logits in enumerate(mtp_logits_list[1:]): # Skip the first one as its loss is already calculated in ce_loss
            # i=0 -> Head predicts 2nd future token (t+2)
            # i=1 -> Head predicts 3rd future token (t+3)
            
            # The 'distance' from the Main Head target is i + 1
            shift = i + 1
            
            mtp_logits = mtp_logits.contiguous()
            slice_logits = mtp_logits[:, :-shift, :].contiguous()
            slice_labels = labels[:, shift:].contiguous()
            
            if slice_labels.size(1) > 0:
                current_mtp_loss = self.loss_fun(
                    slice_logits.view(-1, slice_logits.size(-1)),
                    slice_labels.view(-1)
                )
                mtp_loss_sum += current_mtp_loss
                # We store the weighted component expected for backward
                # The total loss uses average: mtp_lambda * sum / N
                # So component is: mtp_lambda * current_mtp_loss / N
                if self.perform_gradient_projection:
                    N = len(mtp_logits_list) - 1
                    mtp_component_losses.append(self.mtp_lambda * current_mtp_loss / N)
        
        # Combine losses
        mtp_term = self.mtp_lambda * (mtp_loss_sum / (len(mtp_logits_list) - 1))
        total_loss = next_token_loss + mtp_term
        
        if self.perform_gradient_projection:
            return total_loss, next_token_loss, mtp_term, mtp_component_losses
            
        return total_loss, next_token_loss, mtp_term

    def _parse_arguments(
        self,
        args: list,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        
        # Extract MTP logits list
        mtp_logits_list = lm_logits[self.mtp_prediction_key]
        return labels, lm_logits, mtp_logits_list

class MTPCrossEntropyLossTemporalDiscounting(Loss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        mtp_prediction_key: str = "mtp_logits", 
        mtp_lambda: float = 1.0,
        discount_factor: float = 0.9,
        mtp_lambda_scheduler = None,
        perform_gradient_projection: bool = False,
        monitor_gradient_conflicts: bool = False,
        tag: str = "MTPCrossEntropyLossTemporalDiscounting"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve the MAIN head logits.
            mtp_prediction_key: Key to retrieve the list of MTP head logits.
            mtp_lambda: Weighting factor for the MTP auxiliary loss. 
                        Paper suggests 1.0 or similar. This is the initial value
                        if mtp_lambda_scheduler is provided.
            discount_factor: Temporal discount factor for future predictions.
            mtp_lambda_scheduler: Optional scheduler for dynamically adjusting mtp_lambda.
                                   If provided, uses scheduler.mtp_lambda instead of self.mtp_lambda.
            perform_gradient_projection: If True, returns detailed losses to allow gradient projection.
            tag: Tag for logging.
        """
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.mtp_prediction_key = mtp_prediction_key
        self.mtp_lambda = mtp_lambda
        self.discount_factor = discount_factor
        self.mtp_lambda_scheduler = mtp_lambda_scheduler
        self.perform_gradient_projection = perform_gradient_projection
        self.monitor_gradient_conflicts = monitor_gradient_conflicts

        
        # Mean over the tokens in the local-batch
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, mtp_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, _, mtp_logits_list = self._parse_arguments(args, kwargs)

        # Move labels to correct device
        labels = labels.to(mtp_logits_list[0].device).long()
        shift_logits = mtp_logits_list[0].contiguous()
        shift_labels = labels.contiguous().long()
        
        # --- 1. Main Head Loss (Next Token Prediction) ---
        # If input is x_1...x_t, labels should be x_2...x_{t+1}
        
        # regular next-token prediction loss
        # based on the first iteration logits
        next_token_loss = self.loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # --- 2. MTP Heads Loss ---
        # if mtp_logits_list:
        mtp_loss_sum = 0.0
        mtp_component_losses = []
        
        # Use scheduled mtp_lambda if scheduler is provided, otherwise use the fixed value
        current_mtp_lambda = (
            self.mtp_lambda_scheduler.mtp_lambda 
            if self.mtp_lambda_scheduler is not None 
            else self.mtp_lambda
        )

        for i, mtp_logits in enumerate(mtp_logits_list[1:]): # Skip the first one as its loss is already calculated in ce_loss
            # i=0 -> Head predicts 2nd future token (t+2)
            # i=1 -> Head predicts 3rd future token (t+3)
            
            # The 'distance' from the Main Head target is i + 1
            shift = i + 1
            
            mtp_logits = mtp_logits.contiguous()
            slice_logits = mtp_logits[:, :-shift, :].contiguous()
            slice_labels = labels[:, shift:].contiguous()
            
            if slice_labels.size(1) > 0:
                current_mtp_loss = self.loss_fun(
                    slice_logits.view(-1, slice_logits.size(-1)),
                    slice_labels.view(-1)
                )
                weighted_loss = current_mtp_loss * (self.discount_factor ** shift)
                mtp_loss_sum += weighted_loss
                if self.perform_gradient_projection:
                    mtp_component_losses.append(current_mtp_lambda * weighted_loss)
        
        # Combine losses
        mtp_term = current_mtp_lambda * mtp_loss_sum
        total_loss = next_token_loss + mtp_term
        
        if self.perform_gradient_projection:
            return total_loss, next_token_loss, mtp_term, mtp_component_losses
            
        return total_loss, next_token_loss, mtp_term

    def _parse_arguments(
        self,
        args: list,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        
        # Extract MTP logits list
        mtp_logits_list = lm_logits[self.mtp_prediction_key]
        return labels, lm_logits, mtp_logits_list


def nce_loss(
    embedding1: torch.Tensor, embedding2: torch.Tensor, device: torch.device, is_asymmetric: bool, temperature: float
) -> torch.Tensor:
    """
    This implementation calculates the noise contrastive estimation loss between embeddings of two different modalities
    Implementation slightly adapted from https://arxiv.org/pdf/1912.06430.pdf, https://github.com/antoine77340/MIL-NCE_HowTo100M
    changes include adding a temperature value and the choice of calculating asymmetric loss w.r.t. one modality
    This implementation is adapted to contrastive loss from CoCa model https://arxiv.org/pdf/2205.01917.pdf

    Args:
        embedding1 (torch.Tensor): embeddings from modality 1 of size batch_size x embed_dim.
        embedding2 (torch.Tensor): embeddings from modality 2 of size batch_size x embed_dim.
        device (torch.device): torch device for calculating loss.
        is_asymmetric (bool): boolean value to specify if the loss is calculated in one direction or both directions.
        temperature (float): temperature value for regulating loss.

    Returns:
            torch.Tensor: loss tensor.
    """
    # calculating the similarity matrix of size (batch_size x batch_size)
    sim_matrix = torch.matmul(embedding1, embedding2.t()) / temperature
    # numerator of loss: using similarity scores for all positive pairs (e.g., image and its caption)
    numerator = sim_matrix * torch.eye(sim_matrix.shape[0], device=device)
    numerator = numerator.sum(dim=0).view(sim_matrix.shape[0], -1)
    numerator = torch.logsumexp(numerator, dim=1)
    if is_asymmetric:
        # denominator of loss: using all similarity scores for all pairs (positive and negative)
        denominator = torch.logsumexp(sim_matrix, dim=1)
    else:
        # calculate bidirectional loss
        numerator *= 2
        denominator = torch.logsumexp(sim_matrix, dim=1) + torch.logsumexp(sim_matrix.t(), dim=1)
    return torch.mean(denominator - numerator)  # calculated in log space


class NCELoss(Loss):
    def __init__(
        self,
        prediction_key1: str,
        prediction_key2: str,
        is_asymmetric: bool = True,
        temperature: float = 1.0,
        tag: str = "NCELoss",
    ):
        """
        Noise Contrastive Estimation Loss

        Args:
            prediction_key1 (str): key to access embedding 1.
            prediction_key2 (str): key to access embedding 2.
            is_asymmetric (bool, optional): specifies symmetric or asymmetric calculation of NCEloss. Defaults to True.
            temperature (float, optional): temperature. Defaults to 1.0.
            tag (str, optional): Defaults to "NCELoss".
        """
        super().__init__(tag)
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.is_asymmetric = is_asymmetric
        self.temperature = temperature

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch.

        Returns:
            torch.Tensor: loss tensor.
        """
        embedding1 = forward_batch.get_predictions(self.prediction_key1)
        embedding2 = forward_batch.get_predictions(self.prediction_key2)

        contiguous_embedding1 = embedding1.contiguous()
        contiguous_embedding2 = embedding2.contiguous()

        loss = nce_loss(
            contiguous_embedding1, contiguous_embedding2, embedding1.device, self.is_asymmetric, self.temperature
        )
        return loss

class MTPCrossEntropyLossTemporalDiscountingPonder(Loss):
    def __init__(
        self, 
        target_key: str, 
        prediction_key: str, 
        mtp_prediction_key: str = "mtp_logits", 
        mtp_lambda: float = 1.0,
        discount_factor: float = 0.9,
        mtp_lambda_scheduler = None,
        efficiency_lambda_scheduler = None,
        ponder_weight: float = 0.01,
        mixed_gate_loss: bool = False,
        tag: str = "MTPCrossEntropyLossTemporalDiscountingPonder"
    ):
        """
        Args:
            target_key: Key to retrieve targets from the batch.
            prediction_key: Key to retrieve the MAIN head logits.
            mtp_prediction_key: Key to retrieve the list of MTP head logits.
            mtp_lambda: Weighting factor for the MTP auxiliary loss. 
                        Paper suggests 1.0 or similar. This is the initial value
                        if mtp_lambda_scheduler is provided.
            discount_factor: Temporal discount factor for future predictions.
            mtp_lambda_scheduler: Optional scheduler for dynamically adjusting mtp_lambda.
                                   If provided, uses scheduler.mtp_lambda instead of self.mtp_lambda.
            perform_gradient_projection: If True, returns detailed losses to allow gradient projection.
            tag: Tag for logging.
        """
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.mtp_prediction_key = mtp_prediction_key
        self.mtp_lambda = mtp_lambda
        self.mixed_gate_loss = mixed_gate_loss
        self.discount_factor = discount_factor
        self.mtp_lambda_scheduler = mtp_lambda_scheduler
        self.efficiency_lambda_scheduler = efficiency_lambda_scheduler
        self.ponder_weight = ponder_weight

        
        # Mean over the tokens in the local-batch
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, mtp_outputs: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, _, mtp_logits_list, ponder_regularization_loss, gates = self._parse_arguments(args, kwargs)

        # Move labels to correct device
        labels = labels.to(mtp_logits_list[0].device).long()
        shift_logits = mtp_logits_list[0].contiguous()
        shift_labels = labels.contiguous().long()
        
        # --- 1. Main Head Loss (Next Token Prediction) ---
        # If input is x_1...x_t, labels should be x_2...x_{t+1}
        
        # regular next-token prediction loss
        # based on the first iteration logits
        next_token_loss = self.loss_fun(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # --- 2. MTP Heads Loss ---
        # if mtp_logits_list:
        mtp_loss_sum = 0.0
        mtp_component_losses = []
        
        # Use scheduled mtp_lambda if scheduler is provided, otherwise use the fixed value
        current_mtp_lambda = (
            self.mtp_lambda_scheduler.mtp_lambda 
            if self.mtp_lambda_scheduler is not None 
            else self.mtp_lambda
        )

        current_ponder_weight = (
            self.efficiency_lambda_scheduler.mtp_lambda 
            if self.efficiency_lambda_scheduler is not None
            else self.ponder_weight
        )

        for i, mtp_logits in enumerate(mtp_logits_list[1:]): # Skip the first one as its loss is already calculated in ce_loss
            # i=0 -> Head predicts 2nd future token (t+2)
            # i=1 -> Head predicts 3rd future token (t+3)
            
            # The 'distance' from the Main Head target is i + 1
            shift = i + 1
            
            mtp_logits = mtp_logits.contiguous()
            slice_logits = mtp_logits[:, :-shift, :].contiguous()
            slice_labels = labels[:, shift:].contiguous()
            
            if slice_labels.size(1) > 0:
                current_mtp_loss = self.loss_fun(
                    slice_logits.view(-1, slice_logits.size(-1)),
                    slice_labels.view(-1)
                )
                # gates shape: iterations x batch_size x seq_len x dim
                weighted_loss = current_mtp_loss * (self.discount_factor ** shift)
                if self.mixed_gate_loss:  
                    weighted_loss *= gates[shift].mean()
                
                mtp_loss_sum += weighted_loss
    
        
        
        # Combine losses
        if len(mtp_logits_list) == 1:
            mtp_loss_avg = 0.0
        else:
            mtp_loss_avg = mtp_loss_sum / (len(mtp_logits_list) - 1)
        mtp_term = current_mtp_lambda * mtp_loss_avg
        ponder_term = current_ponder_weight * ponder_regularization_loss
        total_loss = next_token_loss + mtp_term + ponder_term
        
            
        return total_loss, next_token_loss, mtp_term, ponder_term
    
    def _parse_arguments(
        self,
        args: list,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        
        # Extract MTP logits list
        mtp_logits_list = lm_logits[self.mtp_prediction_key]
        ponder_regularization_loss = lm_logits["ponder_regularization_loss"]
        # gates = lm_logits["gates"]
        gates = lm_logits["gates_normalized"]
        return labels, lm_logits, mtp_logits_list, ponder_regularization_loss, gates