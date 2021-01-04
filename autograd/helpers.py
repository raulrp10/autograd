from typing import Union, NamedTuple, Callable, get_type_hints, List
import numpy as np
import matplotlib.pyplot as plt

Array = Union[float, list, np.ndarray]

#Helper Functions
def to_numpy_array(value: Array) -> np.ndarray:
    """Function that transform an input to an np.ndarray

    Args:
        value (Array): [It could have a float, list or np.ndarray type]

    Returns:
        np.ndarray: An numpy array
    """
    #validate_input(validate_array, value = value)  
    return value if isinstance(value, np.ndarray) else np.array(value)

def validate_shape(value: np.ndarray) -> np.ndarray:
    """Function to validate shape of an np.ndarray, prevents to have problem on 1 dimension numpy array

    Args:
        value (np.ndarray): An np array with any dimension

    Returns:
        np.ndarray: An np array with at least 2-dimension size
    """
    if (value.ndim == 1):
        return np.atleast_2d(value).T
    else:
        return value

def validate_tensor_input(value: Array) -> np.ndarray:
    """Function that ensures and transform input value into a numpy array with at least 2 dimension

    Args:
        value (Array): It could be float, list or an np.ndarray

    Returns:
        np.ndarray: An numpy array with at least 2-dimension
    """
    return validate_shape(to_numpy_array(value))

def show_data(dataList: List['PrintableData']) -> None:
    """Draw data on 2D

    Args:
        dataList (List[): PrintableData with x, y and marker arguments for draw
    """
    for data in dataList:
        plt.plot(data.x, data.y, data.marker)
    plt.show()


def reduce_dimensionality(data: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Match the shape of an numpy array to another numpy array

    Args:
        data (np.ndarray): The original value of a numpy array
        grad (np.ndarray): The gradient of the value

    Returns:
        np.ndarray: The gradient of data value on the same shape
    """
    ndims_added = grad.ndim - data.ndim
        
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    
    axis = np.argmax(np.abs(np.array(data.shape) - np.array(grad.shape)))
    if (data.shape != grad.shape):
        grad = grad.sum(axis = axis, keepdims=True)
    
    return grad

#NamedTuples
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

class TensorInfo(NamedTuple):
    data: any
    requires_grad: bool
    depends_on: List[Dependency]

class PrintableData():
    def __init__(self, x, y, marker):
        self.x = x
        self.y = y
        self.marker = marker