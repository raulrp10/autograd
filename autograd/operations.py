import numpy as np
from autograd.helpers import Dependency, validate_shape, TensorInfo, reduce_dimensionality
from typing import List, Optional, Tuple

class Operation:
    def forward(self, value: 'Tensor') -> 'Tensor':
        return value

    def backward(self) -> 'Dependency':
        raise NotImplementedError

    def __call__(self, *args) -> 'TensorInfo':
        data, requires_grad = self.forward(*args)
        dependency = self.backward()
        return TensorInfo(data, requires_grad, dependency)

class Neg(Operation):
    def forward(self, value):
        self.value = value

        data = -value.data
        requires_grad = value.requires_grad

        return (data, requires_grad)
    
    def grad_fn(self, value: np.ndarray):
        return -value

    def backward(self):
        depends_on = []
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        return depends_on
    
class Add(Operation):
    def forward(self, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        self.x = x
        self.y = y
        
        data = x.data + y.data
        requires_grad = x.requires_grad or y.requires_grad

        return (data, requires_grad)

    
    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        grad = reduce_dimensionality(self.x.data, grad)
        return grad
    
    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        grad = reduce_dimensionality(self.y.data, grad)
        return grad
    
    def backward(self):
        depends_on: List[Dependency] = []
        if self.x.requires_grad:
            depends_on.append(Dependency(self.x, self.grad_fn1))
        if self.y.requires_grad:
            depends_on.append(Dependency(self.y, self.grad_fn2))
        return depends_on

class Mul(Operation):
    def forward(self, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        self.x = x
        self.y = y
        
        data = x.data * y.data
        requires_grad = x.requires_grad or y.requires_grad

        return (data, requires_grad)
    
    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.y.data
        grad = reduce_dimensionality(self.x.data, grad)
        return grad
    
    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        grad = grad * self.x.data
        grad = reduce_dimensionality(self.y.data, grad)
        return grad
    
    def backward(self):
        depends_on: List[Dependency] = []
        if self.x.requires_grad:
            depends_on.append(Dependency(self.x, self.grad_fn1))
        if self.y.requires_grad:
            depends_on.append(Dependency(self.y, self.grad_fn2))
        return depends_on

class Div(Operation):
    def forward(self, x: 'Tensor', y: 'Tensor') -> 'Tensor':
        self.x = x
        self.y = y
        
        data = x.data / y.data
        requires_grad = x.requires_grad or y.requires_grad

        return (data, requires_grad)
    
    def grad_fn1(self, grad: np.ndarray) -> np.ndarray:
        grad = (grad * self.y.data) / self.y.data**2
        grad = reduce_dimensionality(self.x.data, grad)
        return grad
    
    def grad_fn2(self, grad: np.ndarray) -> np.ndarray:
        grad = (grad * -self.x.data)/ self.y.data**2
        grad = reduce_dimensionality(self.y.data, grad)
        return grad
    
    def backward(self):
        depends_on: List[Dependency] = []
        if self.x.requires_grad:
            depends_on.append(Dependency(self.x, self.grad_fn1))
        if self.y.requires_grad:
            depends_on.append(Dependency(self.y, self.grad_fn2))
        return depends_on
