import pandas as pd
import numpy as np

data1=pd.read_csv('Data Preprocessing/Synthetic-Dataset/EHAM_LIMC.csv')
data1_1 = data1.to_numpy()
processed = []
current=data1_1[0][4]
for x in data1_1:
    if(current!=x[4]):
        df = pd.DataFrame(processed, columns=['time','lat','lon','alt'])
        df.to_csv(current+'.csv', index=False)
        processed=[]
        current=x[4]
    if(x[0]!=x[0] or x[1]!=x[1] or x[2]!=x[2] or x[3]!=x[3]):
        continue
    else:
        processed.append([x[3],x[0],x[1],x[2]])

for i in range(5733):
    data1=pd.read_csv('Data Preprocessing/Synthetic-Dataset/processed_data/TRAJ_'+str(i)+'.csv')
    data1_1 = data1.to_numpy()
    new=[]
    for j in data1_1:
        new.append([j[1],j[2],j[3],j[0]])
    pd.DataFrame(new, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/Synthetic-Dataset/processed_data_2/TRAJ_'+str(i)+'.csv', index=False)
    