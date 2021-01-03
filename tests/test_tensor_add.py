import unittest
import pytest
from torch import tensor
from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_add(self):
        """ Test sum of tensor with the same shape
        """
        tensor1 = Tensor([2, 4, 6, 8], requires_grad=True)
        tensor2 = Tensor([5, 8, 1, 2], requires_grad=True)

        tensor3 = tensor1 + tensor2
        tensor3.backward(Tensor([-1., -2., -3., -4.]))

        assert tensor3.data.tolist() == [7, 12, 7, 10]
        assert tensor1.grad.data.tolist() == [-1, -2, -3, -4]
        assert tensor2.grad.data.tolist() == [-1, -2, -3, -4]

    def test_add_torch(self):
        """ Test sum of tensor of the same shape and compare with torch
        """
        tensor1 = Tensor([2, 4, 6, 8], requires_grad=True)
        tensor2 = Tensor([2, 4, 6, 8], requires_grad=True)
        tensor3 = tensor1 + tensor2 

        torch_tensor1 = tensor([2, 4, 6, 8], requires_grad = True, dtype=float)
        torch_tensor2 = tensor([2, 4, 6, 8], requires_grad = True, dtype=float)
        torch_tensor3 = torch_tensor1 + torch_tensor2

        torch_tensor3.backward(tensor([-1, -2, -3, -4]))
        tensor3.backward(Tensor([-1, -2, -3, -4]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_add(self):
        """ Test sum of tensors with 1 tensor of different dim (broadcasting)
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)  # (2, 3)
        tensor2 = Tensor([7, 8, 9], requires_grad = True)               # (3,)

        tensor3 = tensor1 + tensor2  # shape (2, 3)
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert tensor2.grad.data.tolist() == [2, 2, 2]
    
    def test_broadcast_add_torch(self):
        """ Test sum of tensors with 1 tensor of different dim (broadcasting) and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)
        tensor2 = Tensor([7, 8, 9], requires_grad = True) 
        tensor3 = tensor1 + tensor2  

        torch_tensor1 = tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True, dtype=float)
        torch_tensor2 = tensor([7, 8, 9], requires_grad = True, dtype=float)
        torch_tensor3 = torch_tensor1 + torch_tensor2        

        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_add_diff_shapes(self):
        """ Test sum of tensors with same dim but different shape
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        tensor2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        tensor3 = tensor1 + tensor2
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert tensor2.grad.data.tolist() == [[2, 2, 2]]
        
    def test_broadcast_add_diff_shapes_torch(self):
        """ Test sum of tensors with same dim but different shape and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)    # (2, 3)
        tensor2 = Tensor([[7, 8, 9]], requires_grad = True)               # (1, 3)

        tensor3 = tensor1 + tensor2
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert tensor2.grad.data.tolist() == [[2, 2, 2]]