import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))

# Boundary (15 x 11)
ax.plot([-7.5, 7.5, 7.5, -7.5, -7.5], [0, 0, 11, 11, 0], 'w-', lw=2)
# Key (4.90 x 5.80)
ax.plot([-2.45, -2.45, 2.45, 2.45], [0, 5.8, 5.8, 0], 'w-', lw=2)
# Arc straight sections (0.90 m from sidelines)
ax.plot([-6.6, -6.6], [0, 2.99], 'w-', lw=2)
ax.plot([ 6.6,  6.6], [0, 2.99], 'w-', lw=2)
# 6.75 m arc
t = np.linspace(0, np.pi, 100)
ax.plot(6.75*np.cos(t), 1.575 + 6.75*np.sin(t), 'w-', lw=2)
# Hoop
ax.plot(0, 1.575, 'yo', markersize=8)

# The 6 calibration clicks (exact official coordinates)
pts = {
    1: (-2.45, 0.0),   # key left  @ baseline
    2: ( 2.45, 0.0),   # key right @ baseline
    3: (-2.45, 5.80),  # free-throw left
    4: ( 2.45, 5.80),  # free-throw right
    5: (-6.60, 0.0),   # arc straight left  @ baseline
    6: ( 6.60, 0.0),   # arc straight right @ baseline
}
for n, (x, y) in pts.items():
    ax.plot(x, y, 'ro', markersize=16)
    ax.annotate(str(n), (x, y), color='white', fontsize=10,
                weight='bold', ha='center', va='center')

ax.set_facecolor('#555')
fig.patch.set_facecolor('#555')
ax.invert_yaxis()          # baseline at TOP, same as your video
ax.set_aspect('equal')
ax.set_title("Click these 6 line intersections in the video, in order 1-6")
plt.tight_layout()
plt.show()