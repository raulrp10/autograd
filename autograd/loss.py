from autograd.module import Module
from autograd.tensor import Tensor
from autograd.functions import Log

class Loss(Module):
    def loss_function(self, y_true: Tensor, y_pred: Tensor):
        raise NotImplemented

    def __call__(self, *args) -> Tensor:
        return self.loss_function(*args)

class CrossEntropyBinary(Loss):
    def binary_cross_entropy(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        ''' Entropia cruzada binaria.'''
        return -y_true * Log()(y_pred) - (Tensor(1) - y_true) * Log()(Tensor(1) - y_pred)

    def loss_function(self, y_true: Tensor, y_pred: Tensor) -> float:
        ''' Costo de clasificación binaria para un batch de datos'''
        m = Tensor(y_true.shape[0])
        return self.binary_cross_entropy(y_true, y_pred).mean()

class CrossEntropy(Loss):
    def loss_function(self, y_true, y_pred):
        m = Tensor(y_true.shape[0])
        return (-y_true * Log()(y_pred)).mean()