import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


data = np.loadtxt("detections_parameters.txt")

X = np.arange(0, 10)
Y = np.arange(60, 91)
X, Y = np.meshgrid(Y, X)
Z = data[:,4]   # hit rate
Z = Z.reshape((10,31))

print("min", np.argmin(Z))
print(np.sort(Z.flatten())[:10])
i,j = np.unravel_index(np.argmin(Z), Z.shape)

print("max", np.argmax(Z))
print(np.sort(Z.flatten())[-10:])
i,j = np.unravel_index(np.argmax(Z), Z.shape)

print("Optimal parameters: idx=",i,"quantile=",j+60)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("detections_parameters_3d.pdf")

plt.figure()
plt.imshow(Z, cmap=cm.coolwarm)
plt.colorbar(shrink=0.4)
plt.xlabel("Threshold percentile")
plt.ylabel("Accumulator index")
plt.xticks([0, 5, 10, 15, 20, 25, 30], labels=["0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90"])
plt.yticks([0, 2, 4, 6, 8], labels=["1", "3", "5", "7", "9"])
plt.savefig("detections_parameters_2d.pdf")
