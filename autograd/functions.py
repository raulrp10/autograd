import numpy as np
from autograd.helpers import Dependency, validate_shape
from autograd.tensor import Tensor
from typing import List, Optional, Tuple

class Function:
    def forward(self, value: 'Tensor') -> 'Tensor':
        return value

    def backward(self) -> 'Dependency':
        raise NotImplementedError

    def __call__(self, *args) -> 'TensorInfo':
        data, requires_grad = self.forward(*args)
        dependency = self.backward()
        return Tensor(data, requires_grad, dependency)

class Sigmoid(Function):
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def forward(self, value: 'Tensor') -> 'Tensor':
        self.value = value

        data = self.sigmoid(value.data)
        requires_grad = value.requires_grad

        return (data, requires_grad)
    
    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        z = self.value.data
        return grad * (self.sigmoid(z) * (1 - self.sigmoid(z)))

    def backward(self) -> 'Dependency':
        depends_on = []
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        return depends_on 

class Relu(Function):
    def forward(self, value: 'Tensor') -> 'Tensor':
        self.value = value

        data = np.maximum(0, value.data)
        requires_grad = value.requires_grad

        return (data, requires_grad)
    
    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        data = self.value.data
        data[data <= 0] = 0
        data[data > 0] = 1
        return grad * data

    def backward(self) -> 'Dependency':
        depends_on = []
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        return depends_on

class Tanh(Function):
    def tanh(self, z):
        exp_z = np.exp(z)
        exp_z_neg = np.exp(-z)
        cosh = (exp_z + exp_z_neg)
        sinh = (exp_z - exp_z_neg)
        return sinh / cosh

    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - np.power(self.tanh(self.value.data),2))

    def forward(self, value: 'Tensor') -> Tuple[np.ndarray, bool]:
        self.value = value

        data = self.tanh(self.value.data)
        requires_grad = value.requires_grad

        return (data, requires_grad)

    def backward(self):
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        else:
            depends_on = []
        return depends_on

class Softmax(Function):
    def softmax(self, data: np.ndarray) -> np.ndarray:
        z = data
        axis = 1 if len(z.shape)>1 else 0
        max_value = np.max(z, axis=axis).reshape((-1, 1))
        z = z - max_value
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp, axis = 1, keepdims=True)
        return z_exp / z_sum

    def forward(self, value: 'Tensor') -> Tuple[np.ndarray, bool]:
        self.value = value

        data = self.softmax(self.value.data)
        requires_grad = self.value.requires_grad

        return (data, requires_grad)

    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        m, n = grad.shape
        p = self.softmax(self.value.data)
        tensor1 = np.einsum('ij,ik->ijk', p, p)
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n)) 
        dSoftmax = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dSoftmax, grad)
        return dz
    
    def backward(self) -> List[Dependency]:
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        else:
            depends_on = []
        return depends_on

class Log(Function):
    def forward(self, value: 'Tensor', eps = 1e-20) -> 'Tensor':
        self.value = value
        self.eps = eps

        data = np.log(np.maximum(value.data, eps))
        requires_grad = value.requires_grad

        return (data, requires_grad)

    def grad_fn(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 / np.maximum(self.value.data, self.eps))

    def backward(self):
        depends_on = []
        if self.value.requires_grad:
            depends_on = [Dependency(self.value, self.grad_fn)]
        return depends_on