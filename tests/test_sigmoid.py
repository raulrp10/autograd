import unittest
import pytest
import numpy as np
from torch import tensor, nn
from autograd.tensor import Tensor
from autograd.functions import Sigmoid

class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        """ Test relu activation function for tensors
        """
        tensor1 = Tensor([-100, 0, 10], requires_grad=True)
        tensor2 = Sigmoid()(tensor1)
        
        tensor2.backward(Tensor([1, 1, 1]))

        assert np.round(tensor2.data,2).tolist() == [0, 0.5, 1]
        assert np.round(tensor1.grad.data,2).tolist() == [0, 0.25, 0]

    def test_sigmoid_torch(self):
        """ Test relu activation function for tensor and compare with torch
        """
        tensor1 = Tensor([-2, 4, -6], requires_grad=True)
        tensor2 = Sigmoid()(tensor1)

        tensor_torch1 = tensor([-2, 4, -6], dtype = float, requires_grad=True)
        sigmoid = nn.Sigmoid()
        tensor_torch2 = sigmoid(tensor_torch1)

        tensor2.backward(Tensor([1, 1, 1]))
        tensor_torch2.backward(tensor([1, 1, 1]))
        
        assert np.round(tensor2.data,5).tolist() == np.round(tensor_torch2.data,5).tolist()
        assert np.round(tensor1.grad.data, 5).tolist() == np.round(tensor_torch1.grad.data, 5).tolist()