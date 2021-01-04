from typing import List, Optional, Union
import numpy as np

from autograd.helpers import Array, to_numpy_array, Dependency, TensorInfo
from autograd.operations import Add, Neg, Mul, Div, Sum, Mean, MatMul

def to_tensor(func) -> 'Tensor':
    def transform(*args, **kargs):
        tensor_info = func(*args, **kargs)
        return Tensor(tensor_info.data, tensor_info.requires_grad, tensor_info.depends_on)
    return transform

def validate_tensor(data: any) -> 'Tensor':
    """[Validate the type of the input and transform to tensor]

    Returns:
        [Tensor]: [Tensor with data input as value]
    """
    return data if isinstance(data, Tensor) else Tensor(data)

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
        """[Transform grad on array of zeros]
        """
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
        """[Change the value of a tensor]

        Returns:
            [type]: [Return the negative value of a tensor]
        """
        return Neg()(self)

    def __sub__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor - tensor]

        Returns:
            [Tensor]: [Sub of two tensors]
        """
        value = validate_tensor(value)
        return self + -value

    def __isub__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor-=tensor]

        Returns:
            [type]: [Substract and assign input value]
        """
        value = validate_tensor(value)
        self.data = self.data - value.data
        return self
    
    def __rsub__(self, value: Array) -> 'Tensor':
        """[Override method for support operation float, array, int - tensor]

        Returns:
            [type]: [Substract value on the left]
        """
        return validate_tensor(value) + -self

    @to_tensor
    def __add__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor + tensor]

        Returns:
            [Tensor]: [Sum of two tensors]
        """
        value = validate_tensor(value)
        return Add()(self, value)

    @to_tensor
    def __iadd__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor+=tensor]

        Returns:
            [type]: [Add and assign input value]
        """
        value = validate_tensor(value)
        self.data = self.data + value.data
        return self

    @to_tensor
    def __radd__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation float, array, int + tensor]

        Returns:
            [type]: [Add value on the left to tensor]
        """
        value = validate_tensor(value)
        return Add()(value, self)

    @to_tensor
    def __mul__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor * tensor]

        Returns:
            [Tensor]: [Mult of two tensors]
        """
        value = validate_tensor(value)
        return Mul()(self, value)
    
    @to_tensor
    def __imul__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor*=tensor]

        Returns:
            [type]: [Add and assign input value]
        """
        value = validate_tensor(value)
        self.data = self.data * value.data
        return self

    @to_tensor
    def __rmul__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation float, array, int * tensor]

        Returns:
            [type]: [Add value on the left to tensor]
        """
        value = validate_tensor(value)
        return Mul()(value, self)

    @to_tensor
    def __truediv__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor / tensor]

        Returns:
            [Tensor]: [Div of two tensors]
        """
        value = validate_tensor(value)
        return Div()(self, value)
    
    @to_tensor
    def __itruediv__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor/=tensor]

        Returns:
            [type]: [Add and assign input value]
        """
        value = validate_tensor(value)
        self.data = self.data / value.data
        return self
    
    @to_tensor
    def __rtruediv__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation float, array, int / tensor]

        Returns:
            [type]: [Add value on the left to tensor]
        """
        value = validate_tensor(value)
        return Div()(value, self)
    
    @to_tensor
    def sum(self) -> 'Tensor':
        """[Sum all elements of tensor data]

        Returns:
            [float]: [Sum of all elements of tensor]
        """
        return Sum()(self)
    
    @to_tensor
    def mean(self) -> 'Tensor':
        """[Return the mean of all elements of tensor data]

        Returns:
            float: [Mean of all elements of tensor data]
        """
        return Mean()(self)
    
    @to_tensor
    def __matmul__(self, value: 'Tensor') -> 'Tensor':
        """[Override method for support operation tensor @ tensor]

        Returns:
            [Tensor]: [MatMul of two tensors]
        """
        assert type(value) != 'autograd.tensor.Tensor', 'Only tensor input is allowed'
        return MatMul()(self, value)
