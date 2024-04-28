import uproot
import pandas as pd
from tqdm import tqdm
import os
for batch in tqdm(range(1,241)):
    os.makedirs('newbatch{}'.format(str(batch)))
    data = uproot.open('batch{}/data/data.root'.format(str(batch)))  #use uproot package to read the original files
    for event in range(1000):
        dic1 = data['PmtHit'].arrays()[event]
        dic2 = data['SipmHit'].arrays()[event]
        df1 = pd.DataFrame(dic1.to_list()).drop('PmtId', axis=1)
        df2 = pd.DataFrame(dic2.to_list()).drop('SipmId', axis=1)
        df = pd.concat([df1, df2])
        # Group by DomId and aggregate, sort by t0 within each group
        grouped_df = df.groupby('DomId').agg({'x0': 'first', 'y0': 'first', 'z0': 'first', 't0': 'min', 'e0': 'count'}).rename(columns={'e0': 'charge'})
        grouped_df = grouped_df.sort_values(by='t0', ignore_index=True)
        grouped_df.to_feather('newbatch{}/event_{}.feather'.format(str(batch),str(event)))
