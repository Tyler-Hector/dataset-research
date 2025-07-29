import pandas as pd
import numpy as np
row_delete = []
count=1
for i in range(1,5229):
    data = pd.read_csv('Data Preprocessing/Synthetic-Zurich/processed_data_2/'+str(i)+'.csv')
    pd.DataFrame(data,columns=['lat', 'lon', 'alt', 'time']).to_csv('Data Preprocessing/Synthetic-Zurich/processed_data/TRAJ_'+str(count)+'.csv',index=False)
    count+=1
    if count >5000:
        break
