import pandas as pd
import os
import sys

dir_dict = dict()
dir_dict['20ng'] = 'runs_20ng'
dir_dict['IMDB'] = 'runs_IMDB'

def main():
    # print command line arguments
    arg = sys.argv[1]
    res = pd.DataFrame()
    for dir in os.listdir(dir_dict[arg]):
      for csvfile in os.listdir(dir_dict[arg] + '/' + dir):
        df = pd.read_csv(dir_dict[arg] + '/' + dir + "/" + csvfile)
        res = pd.concat([res, df])
    res.to_csv(arg + 'assembled.csv')

if __name__ == "__main__":
    main()
