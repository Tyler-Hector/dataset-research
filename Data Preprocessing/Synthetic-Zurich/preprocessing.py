import pandas as pd
import numpy as np
count = 0
row_delete = []
count=1
for i in range(21):
    data = np.load('Data Preprocessing/Synthetic-Zurich/raw_data/'+str(i)+'.npy')
    for j in data:
        pd.DataFrame(j,columns=['lat', 'lon', 'alt', 'time']).to_csv('Data Preprocessing/Synthetic-Zurich/processed_data/'+str(count)+'.csv',index=False)
        count+=1
    if count >5000:
        break
#for i in data1_1:
    #if(i[0] != i[0] or i[1] != i[1] or i[2] != i[2] or i[3] != i[3]):
        #print(f"invalid row at {count}")
        #row_delete.append(count)
    #count += 1
#mask = np.ones(len(data1_1), dtype=bool)
#mask[row_delete] = False
#output = data1_1[mask]
#final = pd.DataFrame(output, columns=['lat', 'lon', 'alt', 'time'])
#final.to_csv('processed_data.csv', index=False)