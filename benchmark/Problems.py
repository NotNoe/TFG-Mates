from pydantic import BaseModel
from abc import ABC, abstractmethod
from benchmark.wrappers.heat1d import test_heat1d_case
from benchmark.wrappers.heat2d import test_heat2d_case
from benchmark.wrappers.laplace import test_laplace_case
from benchmark.wrappers.wave1d import test_wave1d_case
from benchmark.wrappers.wave2d import test_wave2d_case

class Problem(BaseModel, ABC):
    equation: str
    device: str
    idx: int
    csv_idx: int

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


class Heat1D(Problem):
    n_x: int
    n_t: int

    def __init__(self, **data):
        data["equation"] = "heat1d"
        super().__init__(**data)

    def solve(self):
        _, metrics = test_heat1d_case(self.device, self.n_x, self.n_t)
        return metrics
    
    def get_params(self):
        return {"n_x": self.n_x, "n_t": self.n_t}

class Heat2D(Problem):
    n_x: int
    n_y: int
    n_t: int

    def __init__(self, **data):
        data["equation"] = "heat2d"
        super().__init__(**data)

    def solve(self):
        _, metrics = test_heat2d_case(self.device, self.n_x, self.n_y, self.n_t)
        return metrics
    def get_params(self):
        return {"n_x": self.n_x, "n_y": self.n_y, "n_t": self.n_t}


class Wave1D(Problem):
    n_x: int
    n_t: int

    def __init__(self, **data):
        data["equation"] = "wave1d"
        super().__init__(**data)

    def solve(self):
        _, metrics = test_wave1d_case(self.device, self.n_x, self.n_t)
        return metrics
    def get_params(self):
        return {"n_x": self.n_x, "n_t": self.n_t}
    
class Wave2D(Problem):
    n_x: int
    n_y: int
    n_t: int

    def __init__(self, **data):
        data["equation"] = "wave2d"
        super().__init__(**data)

    def solve(self):
        _, metrics = test_wave2d_case(self.device, self.n_x, self.n_y, self.n_t)
        return metrics
    def get_params(self):
        return {"n_x": self.n_x, "n_y": self.n_y, "n_t": self.n_t}

class Laplace(Problem):
    n_x: int
    n_y: int

    def __init__(self, **data):
        data["equation"] = "laplace"
        super().__init__(**data)

    def solve(self):
        _, metrics = test_laplace_case(self.device, self.n_x, self.n_y)
        return metrics
    def get_params(self):
        return {"n_x": self.n_x, "n_y": self.n_y}
