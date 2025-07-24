import pandas as pd
import numpy as np
    
for j in range(1,3089):
    new=[]
    df = pd.read_csv('Data Preprocessing/AirLab/processed_csv/'+str(j)+'.csv')
    arr=df.to_numpy()
    print(j)
    for i in arr:
        new.append(i[])
    df.to_csv('Data Preprocessing/AirLab/processed_csv/'+str(j)+'.csv', index=False)

for i in range(1,3089):
    data1=pd.read_csv('Data Preprocessing/AirLab/processed_csv/'+str(i)+'.csv')
    data1_1 = data1.to_numpy()
    new=[]
    time=0
    for j in data1_1:
        new.append([j[2],j[4],j[3],time])
        time+=0.001
    pd.DataFrame(new, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/AirLab/processed_csv_2/TRAJ_'+str(i)+'.csv', index=False)
    