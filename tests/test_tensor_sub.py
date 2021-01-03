import unittest
import pytest
from torch import tensor
from autograd.tensor import Tensor

# @pytest.mark.skip
class TestTensorSub(unittest.TestCase):
    def test_sub(self):
        """Test substraction of tensor with the same shape
        """
        tensor1 = Tensor([2, 4, 6, 8], requires_grad=True)
        tensor2 = Tensor([5, 8, 1, 2], requires_grad=True)
        
        tensor3 = tensor1 - tensor2
        tensor3.backward(Tensor([-1., -2., -3., -4.]))

        assert tensor3.data.tolist() == [-3, -4, 5, 6]
        assert tensor1.grad.data.tolist() == [-1, -2, -3, -4]
        assert tensor2.grad.data.tolist() == [1, 2, 3, 4]

    def test_sub_torch(self):
        """Test substraction of tensor with the same shape and compare with torch
        """
        tensor1 = Tensor([2, 4, 6, 8], requires_grad=True)
        tensor2 = Tensor([5, 8, 1, 2], requires_grad=True)
        tensor3 = tensor1 - tensor2

        torch_tensor1 = tensor([2, 4, 6, 8], dtype = float, requires_grad=True)
        torch_tensor2 = tensor([5, 8, 1, 2], dtype = float, requires_grad=True)
        torch_tensor3 = torch_tensor1 - torch_tensor2

        tensor3.backward(Tensor([-1., -2., -3., -4.]))
        torch_tensor3.backward(tensor([-1., -2., -3., -4.]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_sub(self):
        """ Test sub of tensors with 1 tensor of different dim (broadcasting)
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([7, 8, 9], requires_grad=True)
        
        tensor3 = tensor1 - tensor2
        tensor3.backward(Tensor([[-1, -1, -1], [-1, -1, -1]]))

        assert tensor3.data.tolist() == [[-6, -6, -6], [-3, -3, -3]]
        assert tensor1.grad.data.tolist() == [[-1, -1, -1], [-1, -1, -1]]
        assert tensor2.grad.data.tolist() == [2, 2, 2]

    def test_broadcast_sub_torch(self):
        """ Test sub of tensors with 1 tensor of different dim (broadcasting) and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)
        tensor2 = Tensor([7, 8, 9], requires_grad = True) 
        tensor3 = tensor1 - tensor2  

        torch_tensor1 = tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True, dtype=float)
        torch_tensor2 = tensor([7, 8, 9], requires_grad = True, dtype=float)
        torch_tensor3 = torch_tensor1 - torch_tensor2        

        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()

    def test_broadcast_sub_diff_shapes(self):
        """ Test sub of tensors with same dim but different shape
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        tensor2 = Tensor([[7, 8, 9]], requires_grad=True)
        
        tensor3 = tensor1 - tensor2
        tensor3.backward(Tensor([[-1, -1, -1], [-1, -1, -1]]))

        assert tensor3.data.tolist() == [[-6, -6, -6], [-3, -3, -3]]
        assert tensor1.grad.data.tolist() == [[-1, -1, -1], [-1, -1, -1]]
        assert tensor2.grad.data.tolist() == [[2, 2, 2]]

    def test_broadcast_sub_diff_shapes_torch(self):
        """ Test sub of tensors with 1 tensor of different dim (broadcasting) and compare with torch
        """
        tensor1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True)
        tensor2 = Tensor([[7, 8, 9]], requires_grad = True) 
        tensor3 = tensor1 - tensor2  

        torch_tensor1 = tensor([[1, 2, 3], [4, 5, 6]], requires_grad = True, dtype=float)
        torch_tensor2 = tensor([[7, 8, 9]], requires_grad = True, dtype=float)
        torch_tensor3 = torch_tensor1 - torch_tensor2        

        torch_tensor3.backward(tensor([[1, 1, 1], [1, 1, 1]]))
        tensor3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert tensor3.data.tolist() == torch_tensor3.data.tolist()
        assert tensor1.grad.data.tolist() == torch_tensor1.grad.data.tolist()
        assert tensor2.grad.data.tolist() == torch_tensor2.grad.data.tolist()
