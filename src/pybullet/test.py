"""
File for showcasing the difference between
Euler's method and RTK4
"""
import numpy as np
import matplotlib.pyplot as plt
from math import e

from pyparsing import col

step = 0.2
stop = 20
y_0 = 1

def f(x_):
    return e ** (0.5 * x_)

def dydx(y_):
    return  0.5 * y_


x_values = np.arange(0, stop, step)
y_values = [f(value) for value in x_values]
euler_values = []
rtk_values = []

current_euler_y = y_0
euler_values.append(current_euler_y)
for x in x_values:
    if x != (stop - step):
        current_euler_y += step * dydx(current_euler_y)
        euler_values.append(current_euler_y)

current_rtk_y = y_0
rtk_values.append(current_rtk_y)
for x in x_values:
    if x != (stop - step):
        k1 = dydx(current_rtk_y)
        k2 = dydx(current_rtk_y + (step / 2) * k1)
        k3 = dydx(current_rtk_y +  (step / 2) * k2)
        k4 = dydx(current_rtk_y + step * k3)
        current_rtk_y += (step / 6) * (k1 + (2 * k2) + (2 * k3) + k4)
        rtk_values.append(current_rtk_y)

#plotting
plt.plot(x_values, y_values, color="red") #normal
plt.plot(x_values, euler_values, color="blue")
plt.plot(x_values, rtk_values, color="green")
plt.show()
#plt.savefig("../data/para.png")
