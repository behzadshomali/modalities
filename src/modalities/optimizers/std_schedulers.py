import math

class STDScheduler:
    def __init__(self, start_factor: float, last_step: int = -1):
        self.last_step = last_step
        self.std = start_factor

    def step(self):
        self.last_step += 1
        self.std = self.get_std()
        return self.std

    def get_std(self) -> float:
        raise NotImplementedError
    
class LinearSTDScheduler(STDScheduler):
    def __init__(self, start_factor: float, end_factor: float, total_steps: int):
        super().__init__(start_factor)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps

    def get_std(self):
        t = min(self.last_step / self.total_steps, 1.0)
        return self.start_factor + t * (self.end_factor - self.start_factor)

class CosineSTDScheduler(STDScheduler):
    def __init__(self, start_factor: float, end_factor: float, total_steps: int):
        super().__init__(start_factor)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps

    def get_std(self):
        t = min(self.last_step / self.total_steps, 1.0)
        cos_t = 0.5 * (1 + math.cos(math.pi * t))
        return self.end_factor + (self.start_factor - self.end_factor) * cos_t


# MTP Lambda Schedulers for scheduling the mtp_lambda value in MTP loss functions
class MTPLambdaScheduler:
    """Base class for MTP lambda schedulers that gradually change the mtp_lambda value."""
    
    def __init__(self, start_value: float, last_step: int = -1):
        self.last_step = last_step
        self.mtp_lambda = start_value

    def step(self):
        self.last_step += 1
        self.mtp_lambda = self.get_mtp_lambda()
        return self.mtp_lambda

    def get_mtp_lambda(self) -> float:
        raise NotImplementedError


class LinearMTPLambdaScheduler(MTPLambdaScheduler):
    """Linearly interpolates mtp_lambda from start_value to end_value over total_steps."""
    
    def __init__(self, start_value: float, end_value: float, total_steps: int):
        super().__init__(start_value)
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_mtp_lambda(self):
        t = min(self.last_step / self.total_steps, 1.0)
        return self.start_value + t * (self.end_value - self.start_value)


class CosineMTPLambdaScheduler(MTPLambdaScheduler):
    """Cosine annealing for mtp_lambda from start_value to end_value over total_steps."""
    
    def __init__(self, start_value: float, end_value: float, total_steps: int):
        super().__init__(start_value)
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps

    def get_mtp_lambda(self):
        t = min(self.last_step / self.total_steps, 1.0)
        # Cosine annealing: starts slow, speeds up, then slows down
        cos_t = 0.5 * (1 + math.cos(math.pi * t))
        return self.end_value + (self.start_value - self.end_value) * cos_t
    
class SigmoidMTPLambdaScheduler(MTPLambdaScheduler):
    """Sigmoid schedule for mtp_lambda from start_value to end_value over total_steps."""
    
    def __init__(
        self, 
        start_value: float, 
        end_value: float, 
        total_steps: int, 
        steepness: float = 10.0, 
        cap_value: float = None,
        training_start_portion: float = 0.0,
        training_end_portion: float = 1.0
    ):
        super().__init__(start_value)
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.steepness = steepness
        self.cap_value = cap_value
        self.training_start_portion = training_start_portion
        self.training_end_portion = training_end_portion

    def get_mtp_lambda(self):
        t = min(self.last_step / self.total_steps, 1.0)
        
        # Adjust t based on start and end portions
        if t <= self.training_start_portion:
            # Before the schedule starts
            adjusted_t = 0.0
        elif t >= self.training_end_portion:
             # After the schedule ends
            adjusted_t = 1.0
        else:
             # During the schedule: normalize t to be between 0 and 1 within the window
            denominator = self.training_end_portion - self.training_start_portion
            if denominator > 0:
                adjusted_t = (t - self.training_start_portion) / denominator
            else:
                 # Start and end portion are the same, jump to end
                adjusted_t = 1.0

        # Sigmoid function centered at adjusted_t=0.5
        sigmoid_t = 1 / (1 + math.exp(-self.steepness * (adjusted_t - 0.5)))
        value = self.start_value + sigmoid_t * (self.end_value - self.start_value)
        if self.cap_value is not None:
            value = min(value, self.cap_value)
        return value