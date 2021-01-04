import unittest
import pytest
from torch import tensor
from autograd.tensor import Tensor

# @pytest.mark.skip
class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        """ Test sum of all elements of tensor
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)

        tensor2 = tensor1.sum()
        tensor2.backward()

        assert tensor2.data.tolist() == 6
        assert tensor1.grad.data.tolist() == [1, 1, 1]

    def test_simple_sum_torch(self):
        """ Test sum of all elements of tensor and compare with torch
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)
        tensor2 = tensor1.sum()
        
        tensor_torch1 = tensor([1, 2, 3], dtype = float, requires_grad=True)
        tensor_torch2 = tensor_torch1.sum()

        tensor2.backward()
        tensor_torch2.backward()

        assert tensor2.data.tolist() == tensor_torch2.data.tolist()
        assert tensor1.grad.data.tolist() == tensor_torch1.grad.data.tolist()

    def test_sum_with_grad(self):
        """ Test sum of all elements of tensor, and input data to backward function
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)
        tensor2 = tensor1.sum()

        tensor2.backward(Tensor(3.))

        assert tensor1.grad.data.tolist() == [3, 3, 3]

    def test_sum_with_grad_torch(self):
        """ Test sum of all elements of tensor, and input data to backward function
            and compare with torch
        """
        tensor1 = Tensor([1, 2, 3], requires_grad=True)
        tensor2 = tensor1.sum()

        tensor_torch1 = tensor([1, 2, 3], dtype = float, requires_grad=True)
        tensor_torch2 = tensor_torch1.sum()

        tensor2.backward(Tensor(3.))
        tensor_torch2.backward(tensor(3.))

        assert tensor1.grad.data.tolist() == tensor1.grad.data.tolist()