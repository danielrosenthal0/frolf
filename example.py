import matplotlib.pyplot as plt

from frispy.disc import Disc

disc = Disc(vx=10, theta=-0.2)
result = disc.compute_trajectory()
times = result.times
x, y, z = result.x, result.y, result.z
plt.plot(x,z)
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.show()