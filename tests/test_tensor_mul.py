import unittest
import pytest
import numpy as np
from torch import tensor
from autograd.tensor import Tensor

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        """ Test product between two tensors with the same shape
        """
        tensor1 = Tensor([2, 4, 6], requires_grad=True)
        tensor2 = Tensor([4, 5, 6], requires_grad=True)

        tensor3 = tensor1 * tensor2
        tensor3.backward(Tensor([-1, -2, -3]))

        assert tensor3.data.tolist() == [8, 20, 36]
        assert tensor1.grad.data.tolist() == [-4, -10, -18]
        assert tensor2.grad.data.tolist() == [-2, -8, -18]

    def test_simple_mul_torch(self):
        """ Test product between two tensors with the same shape and compare with torch
        """
        tensor1 = Tensor([2, 4, 6], requires_grad=True)
        tensor2 = Tensor([4, 5, 6], requires_grad=True)
        tensor3 = tensor1 * tensor2

        torch_tensor1 = tensor([2., 4., 6.], dtype=float, requires_grad=True)
        torch_tensor2 = tensor([4., 5., 6.], dtype=float, requires_grad=True)
        torch_tensor3 = torch_tensor1 * torch_tensor2

        tensor3.backward(Tensor([-1, -2, -3]))
        torch_tensor3.backward(tensor([-1, -2, -3]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist() 

    def test_broadcast_mul_diff_dim(self):
        """ Test product of tensors with 1 tensor of different dim (broadcasting)
        """
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([7, 8, 9], requires_grad=True)

        tensor3 = tensor1 * tensor2
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == [[7, 16, 27], [28, 40, 54]]
        assert tensor1.grad.data.tolist() == [[7., 8., 9.], [7., 8., 9.]]
        assert tensor2.grad.data.tolist() == [5., 7., 9.]


    def test_broadcast_mul_diff_dim_torch(self):
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([7, 8, 9], requires_grad=True)
        tensor3 = tensor1 * tensor2

        #Torch tensors
        torch_tensor1 = tensor([[1, 2, 3],[4, 5, 6]], dtype=float, requires_grad=True)
        torch_tensor2 = tensor([7, 8, 9], dtype=float, requires_grad=True)
        torch_tensor3 = torch_tensor2 * torch_tensor1

        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_mul_diff_shape(self):
        """ Test product of tensors with different shapes (broadcasting)
        """
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([[7, 8, 9]], requires_grad=True)

        tensor3 = tensor1 * tensor2
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == [[7, 16, 27], [28, 40, 54]]
        assert tensor1.grad.data.tolist() == [[7., 8., 9.], [7., 8., 9.]]
        assert tensor2.grad.data.tolist() == [[5., 7., 9.]]

    def test_broadcast_mul_diff_shape_torch(self):
        """ Test product of tensors with different shapes and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([[7, 8, 9]], requires_grad=True)
        tensor3 = tensor1 * tensor2

        #Torch tensors
        torch_tensor1 = tensor([[1, 2, 3],[4, 5, 6]], dtype=float, requires_grad=True)
        torch_tensor2 = tensor([[7, 8, 9]], dtype=float, requires_grad=True)
        torch_tensor3 = torch_tensor2 * torch_tensor1

        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()