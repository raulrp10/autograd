from typing import List, Optional, Union
import numpy as np

from autograd.helpers import Array, to_numpy_array, Dependency, TensorInfo
from autograd.operations import Add, Neg, Mul, Div

def to_tensor(func) -> 'Tensor':
    def transform(*args, **kargs):
        tensor_info = func(*args, **kargs)
        return Tensor(tensor_info.data, tensor_info.requires_grad, tensor_info.depends_on)
    return transform

class Tensor:
    def __init__(self,
                data: Array,
                requires_grad: bool = False,
                depends_on: List[Dependency] = None) -> None:
        self._data = to_numpy_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad = Optional['Tensor']

        if self.requires_grad:
            self.zero_grad()
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._data = data
        self.grad = None

    def __repr__(self) -> str:
        return f'Tensor({self.data}, requires_grad={self.requires_grad}, dependency = {len(self.depends_on)})'

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
    
    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "Called backward on non-requires-grad tensor"
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("Grad must be specified for non-0-tensor")
        
        self.grad.data = self.grad.data + grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    # Supported operations
    @to_tensor
    def __neg__(self) -> 'Tensor':
        return Neg()(self)

    def __sub__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor - tensor]

        Returns:
            [Tensor]: [Sub of two tensors]
        """
        return self + -value
        
    @to_tensor
    def __add__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor + tensor]

        Returns:
            [Tensor]: [Sum of two tensors]
        """
        assert type(value) != 'autograd.tensor.Tensor', 'Only tensor input is allowed'
        return Add()(self, value)
        #return _add(self, ensure_tensor(value))

    @to_tensor
    def __mul__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor * tensor]

        Returns:
            [Tensor]: [Mult of two tensors]
        """
        assert type(value) != 'autograd.tensor.Tensor', 'Only tensor input is allowed'
        return Mul()(self, value)
    
    @to_tensor
    def __truediv__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor / tensor]

        Returns:
            [Tensor]: [Div of two tensors]
        """
        return Div()(self, value)