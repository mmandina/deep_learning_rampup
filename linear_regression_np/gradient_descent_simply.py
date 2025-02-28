import numpy as np

x = 5
ALPHA = .1

func = np.poly1d([1,0,0])
der = np.polyder(func)

# for _ in range(20):
#     x=x-ALPHA*der(x)
#     print(x)

x_vals = np.array([-5, -3, -1, 0, 1, 3, 5]) 
y_vals = (x_vals**2)

for _ in range(100):
    gradients = np.gradient(x_vals,y_vals)
    idx = (np.abs(x_vals-x)).argmin()
    print('idx',idx)
    grad=gradients[idx]
    print('grad',grad)
    x=x-ALPHA*grad
    print('x',x)



