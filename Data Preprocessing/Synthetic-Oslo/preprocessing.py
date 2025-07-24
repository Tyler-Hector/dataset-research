import pandas as pd
import numpy as np

data1=pd.read_csv('Data Preprocessing/test/ENGM.csv')
data1_1 = data1.to_numpy()
count = 0
rows_delete=[] #going to put row numbers to delete

for i in data1_1:
    print(count)
    if(i[0]!=i[0] or i[1]!=i[1] or i[2]!=i[2] or i[3]!=i[3]):
        print("invalid row at {count}")
        rows_delete.append(count)
    count+=1

mask = np.ones(len(data1_1),dtype=bool)
mask[rows_delete]=False
output=data1_1[mask]
final=pd.DataFrame(output, columns=['latitude','longitude','altitude','timedelta','flight_id','callsign','icao24','cluster','timestamp'])
final.to_csv('processed_data.csv', index=False)


data1=pd.read_csv('processed_data.csv')
data1_1 = data1.to_numpy()
current =[]
id=data1_1[0][4]
for i in data1_1:
    if i[4]!=id:
        pd.DataFrame(current,columns=['lat','lon','alt','time']).to_csv(id+'.csv')
        current=[]
    current.append([i[0],i[1],i[2],i[3]])
    id=i[4]
current.to_csv(id+'.csv')