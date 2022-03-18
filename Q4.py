import numpy as np
import random
import math

def gradient_x(x, a, i):
  return 2*x-2*a[i]

def gradient_y(y, b, i):
  return 2*y-2*b[i]

def SGD(x, y, a, b, lr, iter, n):
    for k in range (3):
        x = 1
        y = 1
        if k == 0:
            print('Fixed Learning Rate: \n')
        elif k == 1:
            print('Learning Rate = 0.1/(t+1): \n')
        else:
            print('Learning Rate = 0.1/sqrt(t+1): \n')

        for j in range(iter):
            i = random.randint(0, 2*n - 1) # randomly choose i in range 0-199 (0 indexing)
            grad_x = gradient_x(x, a, i) # use random i as sample index
            grad_y = gradient_y(y, b, i)
            print('iteration: ', j+1, 'x: ', x, 'y: ', y)

            if k == 2:
                lr = 0.1/math.sqrt(j+1)
            elif k == 1:
                lr = 0.1/(j+1)

            x = x - (lr * grad_x)
            y = y - (lr * grad_y)
        
        print('\n')

def main():
    n = 100
    a = np.empty(2*n)
    b = np.empty(2*n)

    # set values for a, b -- (ai, bi)=(i/n, -1)...
    for i in range(2*n):
        if i < n:
            a[i] = (i+1)/n # i+1/n to account for zero indexing
            b[i] = -1
        else:
            a[i] = (i+1-n)/n
            b[i] = 1

    iter = 200
    lr = 0.1
    x = 1
    y = 1

    SGD(x, y, a, b, lr, iter, n)

main()