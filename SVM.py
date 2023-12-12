from   sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mySVM

c = mySVM.SVM()
c.Test()

# X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.40)
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
# plt.show()