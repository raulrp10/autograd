import unittest
import pytest
import numpy as np
from torch import tensor
from autograd.tensor import Tensor


class TestTensorDiv(unittest.TestCase):
    def test_simple_div(self):
        """ Test div between two tensors with the same shape
        """
        tensor1 = Tensor([2, 2.5, 3], requires_grad=True)
        tensor2 = Tensor([4, 5, 6], requires_grad=True)

        tensor3 = tensor2 / tensor1
        tensor3.backward(Tensor([-1, -2, -3]))

        assert tensor3.data.tolist() == [2, 2, 2]
        assert tensor1.grad.data.tolist() == [1, 1.6, 2]
        assert tensor2.grad.data.tolist() == [-0.5, -0.8, -1]

    def test_simple_div_torch(self):
        """ Test div between two tensors with the same shape and compare with torch
        """
        tensor1 = Tensor([2, 2.5, 3], requires_grad=True)
        tensor2 = Tensor([4, 5, 6], requires_grad=True)
        tensor3 = tensor2 / tensor1

        torch_tensor1 = tensor([2, 2.5, 3], dtype = float, requires_grad=True)
        torch_tensor2 = tensor([4, 5, 6], dtype = float, requires_grad=True)
        torch_tensor3 = torch_tensor2 / torch_tensor1

        tensor3.backward(Tensor([-1, -2, -3]))
        torch_tensor3.backward(tensor([-1, -2, -3]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_div_diff_dim(self):
        """ Test div of tensors with 1 tensor of different dim (broadcasting)
        """
        tensor1 = Tensor([[1, 3, 4], [1, 3, 4]], requires_grad = True)  # (2, 3)
        tensor2 = Tensor([2, 3, 4], requires_grad = True)               # (3,)

        tensor3 = tensor2 / tensor1  # shape (2, 3)
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == [[2, 1, 1], [2, 1, 1]]
        assert np.round(tensor1.grad.data, 2).tolist() == [[-2, -0.33, -0.25], [-2, -0.33, -0.25]]
        assert np.round(tensor2.grad.data, 2).tolist() == [2, 0.67, 0.5]

    def test_broadcast_div_diff_dim_torch(self):
        """ Test div of tensors with 1 tensor of different dim (broadcasting) and compare with torch
        """
        tensor1 = Tensor([[1, 3, 4], [1, 3, 4]], requires_grad = True)  # (2, 3)
        tensor2 = Tensor([2, 3, 4], requires_grad = True)               # (3,)
        tensor3 = tensor2 / tensor1  # shape (2, 3)

        torch_tensor1 = tensor([[1, 3, 4], [1, 3, 4]], dtype = float, requires_grad = True)  # (2, 3)
        torch_tensor2 = tensor([2, 3, 4], dtype = float, requires_grad = True)               # (3,)
        torch_tensor3 = torch_tensor2 / torch_tensor1  # shape (2, 3)

        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert np.round(tensor1.grad.data, 2).tolist() == np.round(torch_tensor1.grad.data, 2).tolist()
        assert np.round(tensor2.grad.data, 2).tolist() == np.round(torch_tensor2.grad.data, 2).tolist()

    def test_broadcast_div_diff_shape(self):
        """ Test div of tensors with different shapes
        """
        tensor1 = Tensor([[1, 3, 4], [1, 3, 4]], requires_grad = True)  # (2, 3)
        tensor2 = Tensor([[2, 3, 4]], requires_grad = True)               # (3,)

        tensor3 = tensor2 / tensor1  # shape (2, 3)
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == [[2, 1, 1], [2, 1, 1]]
        assert np.round(tensor1.grad.data, 2).tolist() == [[-2, -0.33, -0.25], [-2, -0.33, -0.25]]
        assert np.round(tensor2.grad.data, 2).tolist() == [[2, 0.67, 0.5]]

    def test_broadcast_div_diff_shape_torch(self):
        """ Test div of tensors with different shapes and compare with torch
        """
        tensor1 = Tensor([[1, 3, 4], [1, 3, 4]], requires_grad = True)  # (2, 3)
        tensor2 = Tensor([[2, 3, 4]], requires_grad = True)               # (3,)
        tensor3 = tensor2 / tensor1  # shape (2, 3)

        torch_tensor1 = tensor([[1, 3, 4], [1, 3, 4]], dtype = float, requires_grad = True)  # (2, 3)
        torch_tensor2 = tensor([[2, 3, 4]], dtype = float, requires_grad = True)               # (3,)
        torch_tensor3 = torch_tensor2 / torch_tensor1  # shape (2, 3)

        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert np.round(tensor1.grad.data, 2).tolist() == np.round(torch_tensor1.grad.data, 2).tolist()
        assert np.round(tensor2.grad.data, 2).tolist() == np.round(torch_tensor2.grad.data, 2).tolist()