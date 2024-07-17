import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.exceptions import NotFittedError

class MemoryModel:
    def __init__(self, kernel='linear', memory_cost=1.0):
        self.kernel = kernel
        self.memory_cost = memory_cost
        self.clf = svm.SVC(probability=True, kernel=self.kernel, C=self.memory_cost)
        self.centroids = None
        self.labels = None

    def initialize(self, points):
        self.centroids = points
        self.labels = np.zeros(len(points))

    def remember(self, point_id):
        self.labels[point_id] = 1
        if len(np.unique(self.labels)) > 1:
            self.clf.fit(self.centroids, self.labels)

    def has_visited(self, features):
        if np.max(self.labels) == 0:
            return False 
        elif np.min(self.labels) == 1:
            return True
        else:
            return self.clf.predict_proba(features.reshape(1, -1))[0][1] > 0.5

def generate_points(n=100):
    return np.random.rand(n, 2)

def on_click(event, points, memory_model, ax, fig, xx, yy):
    if event.inaxes is not ax:
        return

    x, y = event.xdata, event.ydata
    distances = np.linalg.norm(points - np.array([x, y]), axis=1)
    point_id = np.argmin(distances)
    
    memory_model.remember(point_id)

    ax.clear()
    ax.scatter(points[:, 0], points[:, 1], c=memory_model.labels, cmap='coolwarm', edgecolor='k')
    
    if len(np.unique(memory_model.labels)) > 1:
        Z = memory_model.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        levels = np.linspace(Z.min(), Z.max(), 3)
        if Z.min() == Z.max():
            levels = [Z.min(), Z.min() + 1e-5, Z.max() + 2e-5]  # Ensure there are distinct levels
        ax.contourf(xx, yy, Z, alpha=0.3, levels=levels, cmap='coolwarm')

    fig.canvas.draw()

def main():
    points = generate_points(20)
    memory_model = MemoryModel(kernel='rbf', memory_cost=20.0)
    memory_model.initialize(points)

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=memory_model.labels, cmap='coolwarm', edgecolor='k')
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, memory_model, ax, fig, xx, yy))
    
    plt.show()

if __name__ == "__main__":
    main()
