import unittest
import pytest
import numpy as np
from torch import tensor, log
from autograd.tensor import Tensor
from autograd.functions import Log

class TestLog(unittest.TestCase):
    def test_log(self):
        """ Test log function for tensors
        """
        tensor1 = Tensor([1, np.e, np.e**2], requires_grad=True)
        tensor2 = Log()(tensor1)

        tensor2.backward(Tensor([1, np.e, np.e**2]))

        assert tensor2.data.tolist() == [0, 1, 2] 
        assert tensor1.grad.data.tolist() == [1, 1, 1]

    def test_log_torch(self):
        """ Test log function for tensors and compare with torch
        """
        tensor1 = Tensor([1, np.e, np.e**2], requires_grad=True)
        tensor2 = Log()(tensor1)

        torch_tensor1 = tensor([1, np.e, np.e**2], dtype = float, requires_grad=True)
        torch_tensor2 = log(torch_tensor1)

        tensor2.backward(Tensor([1, np.e, np.e**2]))
        torch_tensor2.backward(tensor([1, np.e, np.e**2]))
        
        assert torch_tensor2.data.tolist() == tensor2.data.tolist()
        assert np.round(torch_tensor1.grad.data,2).tolist() == np.round(tensor1.grad.data,2).tolist()
