import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection
import numpy as np # Often used for creating data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # '111' means 1x1 grid, first subplot
data1=pd.read_csv('Data Preprocessing/Synthetic-Oslo/processed_data/TRAJ_0.csv')
data1_1= data1.to_numpy()
x_data=[]
y_data=[]
z_data=[]
for i in data1_1:
    z_data.append(i[3])
    x_data.append(i[1])
    y_data.append(i[2])
ax.plot(x_data, y_data, z_data, color='red')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot of Points')
plt.show()
