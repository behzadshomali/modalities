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