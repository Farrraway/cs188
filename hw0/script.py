import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[2, 0], [0, 2]]

x, y = np.random.multivariate_normal(mean, cov, 1000).T
plt.plot(x, y, 'x')
plt.suptitle('[0,0] mean, covariance [[2, 0], [0, 2]]', fontsize=20)
plt.axis('equal')
plt.show()