import pandas as pd
import os
res = pd.DataFrame()
for dir in os.listdir('runs_IMDB'):
  print (dir)
  for csvfile in os.listdir(dir):
    df = pd.open_csv(csvfile)
    res = res + df
res.to_csv('IMDB_.csv')
  
