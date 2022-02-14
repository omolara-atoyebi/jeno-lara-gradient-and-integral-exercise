import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import derivative

def function(x):
    return x**2 + 2

x = np.linspace(-4, 4, 10000)
X, Y = x, function(x)
max_it = 100
threshold = 0.01

def gradient_descent(function, X, max_it, threshold):
    x = X
    index = np.random.randint(0, len(x))
    point = x[index]
    derivative_in_point = derivative(function, point)
    it = 0
    points = []
    points.append(point)
    while np.abs(derivative_in_point) > threshold and it < max_it:
        if derivative_in_point < 0:
            x = x[index:]
            index = np.random.randint(0, len(x))
            point = x[index]
            points.append(point)
            derivative_in_point = derivative(function, point)
        else:
            x = x[0:index+1]
            index = np.random.randint(0, len(x))
            point = x[index]
            points.append(point)
            derivative_in_point = derivative(function, point)
        it += 1
    plt.figure()
    plt.plot(X, Y)
    plt.scatter(np.array(points), function(np.array(points)), c=range(len(points)), cmap=cm.jet)
    plt.title(str(it))
    plt.colorbar()
    plt.show()

gradient_descent(function, X, max_it, threshold)
