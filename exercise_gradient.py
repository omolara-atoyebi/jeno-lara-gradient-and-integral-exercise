import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.misc import derivative


def function(x):
    return -np.sin(x) ** 3


class GradientDescent:
    def __init__(self, f, X, max_it, threshold):
        self.it = None
        self.points = None
        self.function = f
        self.X = X
        self.max_it = max_it
        self.threshold = threshold

    def gradient_descent(self):
        x = self.X
        index = np.random.randint(0, len(x))
        point = x[index]
        derivative_in_point = derivative(function, point)
        self.it = 0
        self.points = [point]
        while np.abs(derivative_in_point) > self.threshold and self.it < self.max_it:
            if derivative_in_point < 0:
                x = x[index:]
                index = np.random.randint(0, len(x))
                point = x[index]
                self.points.append(point)
                derivative_in_point = derivative(function, point)
            else:
                x = x[0:index + 1]
                index = np.random.randint(0, len(x))
                point = x[index]
                self.points.append(point)
                derivative_in_point = derivative(function, point)
            self.it += 1
        return

    def plot_gradient(self):
        plt.figure()
        plt.plot(self.X, function(self.X))
        plt.scatter(np.array(self.points), function(np.array(self.points)), c=range(len(self.points)), cmap=cm.jet)
        plt.title(str(self.it))
        plt.colorbar()
        plt.show()


x = np.linspace(-4, 4, 10000)

fx = GradientDescent(function(x), x, 100, 0.01)
fx.gradient_descent()
fx.plot_gradient()
