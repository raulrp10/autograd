# Autograd
Trabajo final EDA - Maestria en Ciencía de la Computación

Alumnos:
- Ledgard Raúl Rondon Ponce
- Diego Armando Jacinto Corimanya Huamani

## Requisitos
----
Se necesita python 3.6 o superior para poder correr autograd. 

En ubuntu se puede instalar Python 3:

    $ sudo apt get install python3 python3-pip

Para otros sistemas operativos, se puede encontrar un instalador en:
  
https://www.python.org/getit/

### Virtualenv
----
Se recomienda hacer uso de un entorno virtual ([`virtualenv`](https://virtualenv.pypa.io/en/stable/)).

    $ python3 -m pip install -U virtualenv
    $ python3 -m virtualenv env
    $ source env/bin/activate
    $ pip install -r requirements.txt

## Estructura del proyecto
-----
Para poder ejecutar el proyecto debe instalar las dependencias que se encuentran en el archivo `requirements.txt`

### Autograd
-----
- functions.py: Contiene la implementación de las funciones de activación de las redes neuronales (tanh, relu, sigmoid) y log.
- helpers.py: Contiene funciones y estructuras que sirven de ayuda para la implementación de los tensores.
- loss.py: Implementación de las funciones de costo (SGD).
- module.py: Clase que permite obtener todos los atributos de una clase y devuelve un generador.
- operations.py: Contiene las operaciones que puede realizar un tensor, sirve para la sobrecarga de operadores matemáticos en el tensor.
- parameters.py: Sirve para la generación de los pesos (w) y bias (b), y permite la creación de tensores a tráves de herencia.

### Samples
-----
Para ejecutar alguno de los ejemplos, puede realizarlo de la siguiente manera:

    $ source env/bin/activate
    $ python3 samples/binary_classification.py

- binary_classification.py: Contiene un ejemplo de una red neuronal de 2 capas para un dataset que contiene solamente 2 clases, al final de la clasificación se puede ver visualmente el resultado de la misma.
- linear_regression.py: Contiene un ejemplo de una regresión lineal.
- multiclass_classification.py: Contiene una red neuronal para la clasificación de un dataset de 3 clases.

### Tests
-----
Contiene la implementación de test para probar el correcto funcionamiento de las implementaciones realizadas y compararlas con los resultados obtenidos por pytorch.

Para poder ejecutar todos los test, puede realizar:

    $ pytest

Si se desea solamente ejecutar alguno:

    $ pytest tests/test_tensor_add.py

## Ejemplo de uso
``` python
>>> from autograd.tensor import Tensor
>>> 
>>> data = Tensor([1, 3, 4], requires_grad = True)
>>> sum_data = data + 1
>>> print(sum_data)
Tensor([2, 4, 5], requires_grad=True, dependency = 1)
>>> sum_data.backward([3, 4, 5])
>>> print(data.grad)
Tensor([3, 4, 5], requires_grad=False, dependency = 0)
```

## Referencias
-----
1. http://ace.cs.ohio.edu/~razvan/courses/mlds18/exercises/classification/lr/pytorch.ag/utils.py
2. https://deepnotes.io/softmax-crossentropy
3. https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
4. https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95
5. https://learnml.today/making-backpropagation-autograd-mnist-classifier-from-scratch-in-Python-5
6. https://github.com/joelgrus/autograd/
7. https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
8. https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
9. https://medium.com/ai%C2%B3-theory-practice-business/a-beginners-guide-to-numpy-with-sigmoid-relu-and-softmax-activation-functions-25b840a9a272
10. https://themaverickmeerkat.com/2019-10-23-Softmax/
11. https://www.mathsisfun.com/calculus/derivatives-introduction.html
12. http://cs231n.stanford.edu/handouts/derivatives.pdf
13. https://neuralrepo.net/2020/01/05/chapter-5-vectorized-backpropagation/
14. https://www.jasonosajima.com/backprop