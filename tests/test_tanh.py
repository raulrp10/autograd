import unittest
import pytest
import numpy as np
from autograd.tensor import Tensor
from autograd.functions import Tanh
from torch import nn, tensor

class TestTanh(unittest.TestCase):
    def test_tanh(self):
        """ Test tanh activation functions for tensors
        """
        tensor1 = Tensor([0, 1, 2], requires_grad=True)
        tensor2 = Tanh()(tensor1)

        tensor2.backward(Tensor([1, 1, 1]))

        assert np.round(tensor2.data,2).tolist() == [0, 0.76, 0.96]
        assert np.round(tensor1.grad.data,2).tolist() == [1, 0.42, 0.07]

    def test_tanh_torch(self):
        """ Test tanh activation function for tensor and compare with torch
        """
        tensor1 = Tensor([0, 1, 2], requires_grad=True)
        tensor2 = Tanh()(tensor1)

        tensor_torch1 = tensor([0, 1, 2], dtype = float, requires_grad=True)
        tanh = nn.Tanh()
        tensor_torch2 = tanh(tensor_torch1)

        tensor2.backward(Tensor([1, 1, 1]))
        tensor_torch2.backward(tensor([1, 1, 1]))
        
        assert np.round(tensor2.data,5).tolist() == np.round(tensor_torch2.data,5).tolist()
        assert np.round(tensor1.grad.data, 5).tolist() == np.round(tensor_torch1.grad.data, 5).tolist()
