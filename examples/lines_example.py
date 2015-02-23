#!/usr/bin/python
from matplotlib import pyplot as plt
from lines import *
import numpy as np


l = Lines()
plt.figure()

axis = [-1.0, 1.0, -1.0, 1.0]

'''
Plotting lines.
'''
for line in l.linesbpt:
    plt.plot(l.x[line], l.y[line], label = line)

plt.legend()
plt.title('BPT Lines')
plt.axis(axis)
plt.show()

'''
Generate a random sample.
'''
sigma = 0.50
mu = 1.0
n = 10000
y = sigma * np.random.randn(n) + mu
x = np.linspace(-1.0, 1.0, n)
plt.scatter(x, y)
plt.title('Sample')
plt.axis(axis)
plt.show()

'''
Points above K01.
'''
m = l.maskAbovelinebpt('K01', x, y)
plt.scatter(x[m], y[m], label = 'above K01')
plt.plot(l.x['K01'], l.y['K01'], label = 'K01')
plt.legend()
plt.title('Selecting sample ex. 1')
plt.axis(axis)
plt.show()

'''
Points below S06.
'''
m = l.maskBelowlinebpt('S06', x, y)
plt.scatter(x[m], y[m], label = 'below S06')
plt.plot(l.x['S06'], l.y['S06'], label = 'S06')
plt.legend()
plt.title('Selecting sample ex. 2')
plt.axis(axis)
plt.show()

'''
Points above K01 and below K06
'''
m = l.maskAbovelinebpt('K01', x, y) & l.maskBelowlinebpt('K06', x, y)
plt.scatter(x[m], y[m], label = 'Above K01 and below K06')
plt.plot(l.x['K01'], l.y['K01'], label = 'K01')
plt.plot(l.x['K06'], l.y['K06'], label = 'K06')
plt.legend()
plt.title('Selecting sample ex. 3')
plt.axis(axis)
plt.show()

# vim: set et ts=4 sw=4 sts=4 tw=80 fdm=marker:
