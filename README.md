# udacity-pytorch-deeplearning


```
PyTorch
PyTorch is an open-source Python framework from the Facebook AI Research team used for developing deep neural networks. 
I like to think of PyTorch as an extension of Numpy that has some convenience classes for defining neural networks and accelerated computations using GPUs. 
PyTorch is designed with a Python-first philosophy, it follows Python conventions and idioms, and works perfectly alongside popular Python packages
```


- Introduction:
```
Neural network
Deep Neural network
```

First Quiz:
[https://www.youtube.com/watch?v=X-uMlsBi07k](https://www.youtube.com/watch?v=X-uMlsBi07k)
```
Now that you know the equation for the line (2x1 + x2 - 18=0), and similarly the “score” (2x1 + x2 - 18), what is the score of the student who got 7 in the test and 6 for grades?
```

More Dimension
[https://www.youtube.com/watch?v=eBHunImDmWw](https://www.youtube.com/watch?v=eBHunImDmWw)

Second Quiz

```
Given the table in the video above, what would the dimensions be for input features (x), the weights (W), and the bias (b) to satisfy (Wx + b)?
```

Perceptron Trick
```python
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
        
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

```

Error Function
- error func should be continuous
- error func should be differntiable


Quiz:
The sigmoid function is defined as sigmoid(x) = 1/(1+e-x). If the score is defined by 4x1 + 5x2 - 9 = score, then which of the following points has exactly a 50% probability of being blue or red? (Choose all that are correct.)



```python
>>> a
[(1, 1), (2, 4), (5, -5), (-4, 5)]
>>> import math
>>> sigmoid = lambda x : 1/(1 + math.exp(-x))

>>> for x1,x2 in a:
...     score = 4*x1 + 5*x2 -9
...     print( score, sigmoid(score))
... 
0 0.5
19 0.9999999943972036
-14 8.315280276641321e-07
0 0.5
```


Softmax:
Classification problem

P(gift) = 0.8
P(no gift) = 1 - P(gift) = 0.2

Softmax

a ---> 2
b ---> 1
c ---> 0


p(a) = 2 / (2+1+0)
p(b) = 1 / (2+1+0)
p(c) = 0 / (2+1+0)

The problem is what if we got negative numbers as score.
then it will leads to something/0

Can we fix it ?
import math
math.exp ( ? ) 

cool right ?

a ---> 2
b ---> 1
c ---> 0

p(a) = e^2 / (e^2+ e^1+ e^0)
p(b) = e^1 / (e^2+ e^1+ e^0)
p(c) = e^0 / (e^2+ e^1+ e^0)

This is called softmax ...


QUIZ:

```python
import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    pass
```

Solution

```python
import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    
    eL = np.sum(np.exp(L))
    return [ np.exp(item) / eL for item in L ]
```