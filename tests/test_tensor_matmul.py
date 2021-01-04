import unittest
import pytest
import numpy as np
from torch import tensor
from autograd.tensor import Tensor

# @pytest.mark.skip
class TestTensorMatMul(unittest.TestCase):
    def test_matmul(self):
        """ Test matmul operation between tensors
        """
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([7, 8, 9], requires_grad = True)

        tensor3 = tensor1 @ tensor2
        tensor3.backward(Tensor([2, 1]))
    
    def test_matmul_torch(self):
        """ Test matmul operation between tensors and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([7, 8, 9], requires_grad = True)
        tensor3 = tensor1 @ tensor2

        tensor_torch1 = tensor([[1, 2, 3],[4, 5, 6]], dtype=float, requires_grad=True)
        tensor_torch2 = tensor([7, 8, 9], dtype=float, requires_grad=True)
        tensor_torch3 = tensor_torch1 @ tensor_torch2

        tensor3.backward(Tensor([2, 1]))
        tensor_torch3.backward(tensor([2, 1]))

        assert tensor3.data.tolist() == tensor_torch3.data.tolist()
        assert tensor1.grad.data.tolist() == tensor_torch1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == tensor_torch2.grad.data.tolist()
    