# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import traits

def parse_tvn():
    #folder = 'Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Age-test_acc-0.8619641547007652.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Ethnicity-test_acc-0.7924745833770045.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Gender-test_acc-0.8691961010376271.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Others-test_acc-0.6660727387066345.csv'
    file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Religion-test_acc-0.9057750759878419.csv'


    df = pd.read_csv(file)
    print(len(df))
    print(df.columns)
    # Number in per label that are correct
    df['match'] = df['target']==df['y_pred']
    #print(df)
    
    #unmatched = pd.get_dummies(np.where(df['target']==df['y_pred'],'matched','unmatched')).sum(0)

    #print(df[0:4]) 

    # # Number in per label that are correct
    # # also size(), count(), nth(), last()
    count = df.groupby('label').size()
    print(count)

    #df_group = df.groupby("label")
    # df_group.get_group('Age')


    # for name_of_group, contents_of_group in df_group:
    #     print(name_of_group)
    #     print(contents_of_group)
    
    # df.groupby('var1')['var2'].apply(lambda x: (x=='val').sum()).reset_index(name='count')

    # df_groupby = df.groupby('label')['match'].apply(lambda x: (x==True).sum()).reset_index(name='count')
    df_count = df.groupby('label')['match'].apply(lambda x: (x==True).sum()).reset_index(name='count')
    print(df_count)
    print(type(df_count))

if __name__=="__main__":
    parse_tvn()