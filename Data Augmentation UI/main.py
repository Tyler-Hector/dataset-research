import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# Load the data
data = np.loadtxt(r'C:\Users\debas\AI trajectory\dataset-research\DataSet1\day1\7days1\processed_data\test\2.txt')

# Extract columns
x = data[:, 2]  # Column 3 (zero-indexed)
y = data[:, 1]  # Column 2
z = data[:, 3]  # Column 4

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z)

# Hide tick labels
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()
