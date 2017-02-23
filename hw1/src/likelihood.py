import matplotlib.pyplot as plt
import numpy as np

def likelihood_across_p(n, n_ones):
	X = np.linspace(0, 1.0, num=100)
	y = []
	for theta in X:
		likelihood = (theta ** n_ones) * ((1 - theta) ** (n - n_ones))
		print(likelihood)
		y.append(likelihood)
	return X, y



# X, y = likelihood_across_p(5, 3)
# scatter1 = plt.scatter(X, y, c='r', s=20)

X, y = likelihood_across_p(100, 60)
scatter2 = plt.scatter(X, y, c='c', s=20)
# set axes range
plt.ylim(0, 0.00000000000000000000000000001)

X, y = likelihood_across_p(10, 5)
# scatter3 = plt.scatter(X, y, c='y', s=20)

plt.suptitle('Maximum Likelihood Estimator 60/100 ones', fontsize=20)
plt.xlabel('probability', fontsize=16)
plt.ylabel('likelihood', fontsize=16)
plt.show()

# , '60/100 ones', '5/10 ones'