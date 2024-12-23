import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


 #Visualize the decision boundary
def plot_decision_boundary(X, y, model, ax, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.RdBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k', marker='o')
    ax.set_title(title)
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')

    return scatter

   #Plot the decision boundary
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_boundary(X_train, y_train, svm, ax, "SVM with Linear Kernel")
plt.show()
