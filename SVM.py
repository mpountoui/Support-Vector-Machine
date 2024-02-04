from sklearn.datasets import make_blobs
from sklearn          import svm

import matplotlib.pyplot as plt
import numpy             as np
import mySVM

#X, Y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.40)

X = np.array([[1, 5], [2, 5], [3, 8], [4, 3], [5, 10], [6, 1]])
Y = np.array([-1, -1, 1, 1, 1, -1])  # Class labels: 0 for apples, 1 for oranges

for i in range(len(Y)):
    if(Y[i] == 0):
        Y[i] = -1

#Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Weight')
plt.ylabel('Sweetness')
plt.title('Toy Dataset: Apples vs Oranges')
plt.show()

#clf = svm.SVC(kernel='rbf', C = 2.5, gamma=1.0)
clf = mySVM.SVM()

clf.fit(X, Y)

xx, yy = np.meshgrid( np.linspace(min(X[:, 0]), max(X[:, 0]) , 100), 
                      np.linspace(min(X[:, 1]), max(X[:, 1]) , 100) )
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize the data and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlabel('Weight')
plt.ylabel('Sweetness')
plt.title('SVM Classification')
plt.show()
