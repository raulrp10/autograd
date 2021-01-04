import unittest
import pytest
import numpy as np
from torch import tensor
from autograd.tensor import Tensor

# @pytest.mark.skip
class TestTensorSum(unittest.TestCase):
    def test_simple_mean(self):
        """ Test mean of all elements of tensor
        """
        tensor1 = Tensor([3, 6, 9], requires_grad=True)

        tensor2 = tensor1.mean()
        tensor2.backward()

        assert tensor2.data.tolist() == 6
        assert np.round(tensor1.grad.data,2).tolist() == [0.33, 0.33, 0.33]

    def test_simple_mean_torch(self):
        """ Test mean of all elements of tensor and compare with torch
        """
        tensor1 = Tensor([3, 6, 9], requires_grad=True)
        tensor2 = tensor1.mean()
        
        tensor_torch1 = tensor([3, 6, 9], dtype = float, requires_grad=True)
        tensor_torch2 = tensor_torch1.mean()

        tensor2.backward()
        tensor_torch2.backward()

        assert tensor2.data.tolist() == tensor_torch2.data.tolist()
        assert tensor1.grad.data.tolist() == tensor_torch1.grad.data.tolist()

    def test_mean_with_grad(self):
        """ Test mean of all elements of tensor, and input data to backward function
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)
        tensor2 = tensor1.mean()

        tensor2.backward(Tensor(3.))

        assert tensor1.grad.data.tolist() == [1, 1, 1]

    def test_mean_with_grad_torch(self):
        """ Test mean of all elements of tensor, and input data to backward function
            and compare with torch
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)
        tensor2 = tensor1.mean()

        tensor_torch1 = tensor([1, 2, 3], dtype = float, requires_grad=True)
        tensor_torch2 = tensor_torch1.mean()

        tensor2.backward(Tensor(3.))
        tensor_torch2.backward(tensor(3.))

        assert tensor1.grad.data.tolist() == tensor1.grad.data.tolist()