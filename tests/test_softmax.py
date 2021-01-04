import unittest
import pytest
import numpy as np
from torch import nn, tensor
from autograd.tensor import Tensor
from autograd.functions import Softmax

class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        """ Test relu activation function for tensors
        """
        tensor1 = Tensor([[-2, 4, -6]], requires_grad=True)
        tensor2 = Softmax()(tensor1)
        
        tensor2.backward(Tensor([[1, 1, 1]]))

        assert np.round(tensor2.data,2).tolist() == [[0, 1, 0]]
        assert np.round(tensor1.grad.data,2).tolist() == [[0, 0, 0]]

    def test_softmax_torch(self):
        """ Test relu activation function for tensor and compare with torch
        """
        tensor1 = Tensor([[-2, 4, -6]], requires_grad=True)
        tensor2 = Softmax()(tensor1)

        tensor_torch1 = tensor([[-2, 4, -6]], dtype = float, requires_grad=True)
        softmax = nn.Softmax()
        tensor_torch2 = softmax(tensor_torch1)

        tensor2.backward(Tensor([[1, 1, 1]]))
        tensor_torch2.backward(tensor([[1, 1, 1]]))
        
        assert np.round(tensor2.data,5).tolist() == np.round(tensor_torch2.data,5).tolist()
        assert np.round(tensor1.grad.data, 5).tolist() == np.round(tensor_torch1.grad.data, 5).tolist()