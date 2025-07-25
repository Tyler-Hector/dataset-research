import numpy as np
import pandas as pd
import random

def augmentation (name):
    print(name)
    count=[3088,5731,6272]
    type=-1
    if name=='AirLab':
        type=0
    elif name=='Synthetic-Dataset':
        type=1
    elif name=='Synthetic-Oslo':
        type=2
    flip=[]
    move=[]
    combo=[]
    rand_x=random.uniform(-50.0, 50.0)
    rand_y=random.uniform(-50.0, 50.0)
    for i in range(1,count[type]+1):
        print(i)
        data=pd.read_csv('Data Preprocessing/'+name+'/processed_data/TRAJ_'+str(i)+'.csv', engine='c').to_numpy()
        for j in data:
            flip.append([j[1],j[0],j[2],j[3]])
            move.append([j[0]+rand_x,j[1]+rand_y,j[2],j[3]])
            combo.append([j[1]+rand_x,j[0]+rand_y,j[2],j[3]])
        pd.DataFrame(data, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/final/original_'+name+'_'+str(i)+'.csv', index=False)
        pd.DataFrame(flip, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/final/flip_'+name+'_'+str(i)+'.csv', index=False)
        pd.DataFrame(move, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/final/move_'+name+'_'+str(i)+'.csv', index=False)
        pd.DataFrame(combo, columns=['lat','lon','alt','time']).to_csv('Data Preprocessing/final/combo_'+name+'_'+str(i)+'.csv', index=False)
    

augmentation('AirLab')
augmentation('Synthetic-Dataset')
augmentation('Synthetic-Oslo')
