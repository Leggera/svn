import pandas as pd
import os
res = pd.DataFrame()
for dir in os.listdir('runs_IMDB'):
  for csvfile in os.listdir('runs_IMDB/' + dir):
    df = pd.read_csv('runs_IMDB/' + dir + "/" + csvfile)
    res = res + df
res.to_csv('IMDB_.csv')
  
