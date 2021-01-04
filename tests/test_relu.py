import unittest
import pytest
import numpy as np
from autograd.tensor import Tensor
from autograd.functions import Relu
from torch import tensor, nn

class TestRelu(unittest.TestCase):
    def test_relu(self):
        """ Test relu activation function for tensors
        """
        tensor1 = Tensor([-2, 4, -6], requires_grad=True)
        tensor2 = Relu()(tensor1)
        
        tensor2.backward(Tensor([1, 1, 1]))

        assert tensor2.data.tolist() == [0, 4, 0]
        assert np.round(tensor1.grad.data,2).tolist() == [0, 1, 0]

    def test_relu_torch(self):
        """ Test relu activation function for tensor and compare with torch
        """
        tensor1 = Tensor([-2, 4, -6], requires_grad=True)
        tensor2 = Relu()(tensor1)

        tensor_torch1 = tensor([-2, 4, -6], dtype = float, requires_grad=True)
        relu = nn.ReLU()
        tensor_torch2 = relu(tensor_torch1)

        tensor2.backward(Tensor([1, 1, 1]))
        tensor_torch2.backward(tensor([1, 1, 1]))
        
        assert np.round(tensor2.data,5).tolist() == np.round(tensor_torch2.data,5).tolist()
        assert np.round(tensor1.grad.data, 5).tolist() == np.round(tensor_torch1.grad.data, 5).tolist()
