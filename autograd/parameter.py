import numpy as np

from autograd.tensor import Tensor

np.random.seed(1)

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape) * 0.01
        super().__init__(data, requires_grad=True)