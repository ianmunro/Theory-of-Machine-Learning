import numpy as np

# 1D array from -50 to 50, no x_i = 0
x = np.arange(-50, 51)
x = np.delete(x, 50)
y = np.concatenate((np.repeat(-1, 50), np.repeat(1, 50)), axis=0)

def Gradient(x, y, w):
  return np.sum((-y*x*np.exp(-y*x*w))/(1+np.exp(-y*x*w))) # gradient of L(w)

def GradientDescent(x, y, w, lr, iter):
    for i in range(iter):
        grad = Gradient(x, y, w)
        print('iteration: ', i, 'gradient: ', grad, 'w: ', w)
        w = w - (lr * grad)

def Problem1C():
    print('Problem 1(c):\n')    
    x = np.arange(-50, 51) # 1D array from -50 to 50, 
    x = np.delete(x, 50) # no x_i = 0
    y = np.concatenate((np.repeat(-1, 50), np.repeat(1, 50)), axis=0)

    w = -1
    iter = 100
    lr = 1
    GradientDescent(x, y, w, lr, iter)

def Problem1D():
    print('\nProblem 1(d):\n')   
    x = np.arange(-50, 51) # 1D array from -50 to 50, 
    x = np.delete(x, 50) # no x_i = 0
    y = np.concatenate((np.repeat(-1, 50), np.repeat(1, 50)), axis=0)

    y_corrupt = y
    length = len(y_corrupt)
    for i in range(5):
        y_corrupt[i] = -y_corrupt[i] # swap signs for first 5 elements
        y_corrupt[length - i - 1] = -y_corrupt[i] # swap signs of last 5 elements

    w = -1
    iter = 100
    lr = 0.00005
    GradientDescent(x, y_corrupt, w, lr, iter)

def main():
    Problem1C()
    Problem1D()

main()