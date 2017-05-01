import pandas as pd
import os
res = pd.DataFrame()
for dir in os.listdir('runs_20ng'):
  for csvfile in os.listdir('runs_20ng/' + dir):
    df = pd.read_csv('runs_20ng/' + dir + "/" + csvfile)
    res = pd.concat([res, df])
res.to_csv('20ng_.csv')
  
