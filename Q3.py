import numpy as np
import math
import timeit

def Gradient(A, x, b):
  return 2*np.dot(A.T, (np.dot(A,x)-b))

def GradientDescent(A, x, b, lr, iter):
    for i in range(iter):
        grad = Gradient(A, x, b)
        x = x - (lr * grad)
    return x # save the weights

def main():
    #### c) random dataset ####
    n = 500
    m = 2*n
    A = (np.random.rand(m, n) * 2) - 1 # range [0,1] * 2 = [0, 2], - 1 = [-1, 1]
    x_star = (np.random.rand(n) * 2) - 1
    eta = np.random.normal(loc = 0, scale = math.sqrt(0.5), size=m)
    b = np.dot(A, x_star) + eta
    ########

    #### d) gradient descent ####
    lr = 0.001
    iter = 50
    x = np.zeros(n)

    x = GradientDescent(A, x, b, lr, iter)

    f_x = (np.linalg.norm(np.dot(A, x)-b, ord=2))**2
    print('Function value: ', f_x, '\n')
    dist = x_star - x
    print('Distance from x*:\n', dist)
    ########

    #### e) running time closed form ####
    start = timeit.default_timer()
    GradientDescent(A, x, b, lr, iter)
    stop = timeit.default_timer()
    print('\nGradient descent running time: ', stop - start, '\n')

    start = timeit.default_timer()
    closed_form = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), b)
    stop = timeit.default_timer()
    print('Closed form running time: ', stop - start)
    ########


main()