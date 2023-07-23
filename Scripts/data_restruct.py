# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import os,sys
import pandas as pd

six = ['../Dataset/SixClass/test.csv', '../Dataset/SixClass/train.csv', '../Dataset/SixClass/valid.csv']
num_trt = 6

for flnm in six:
    six_df = pd.read_csv(flnm)
    print('file shape is ', six_df.shape)
    #print(six_df.head())
    for trt in range(num_trt):
        six_trait=six_df.loc[six_df['target'] == trt]
        print('trait ', trt, 'shape is ',six_trait.shape)
        #print(six_trait.head())
        print()



