# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import traits
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def parse_tvn():
    #folder = 'Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Age-test_acc-0.8619641547007652.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Ethnicity-test_acc-0.7924745833770045.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Gender-test_acc-0.8691961010376271.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Others-test_acc-0.6660727387066345.csv'
    file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Religion-test_acc-0.9057750759878419.csv'


    df = pd.read_csv(file)
    total_all = len(df)
    print(total_all)
    print(df.columns)
    # Number in per label that are correct
    df['match'] = df['target']==df['y_pred']
    #print(df)
    
    #unmatched = pd.get_dummies(np.where(df['target']==df['y_pred'],'matched','unmatched')).sum(0)

    #print(df[0:4]) 

    # # Number in per label that are correct
    # # also size(), count(), nth(), last()
    count = df.groupby('label').size()
    print("count is of type ", type(count))
    print("keys are ", count.axes)
    print("get value for Age ", count.get('Age'))
    print("Size of each trait \n", count)

    #df_group = df.groupby("label")
    
        
    print("confusion matrix for ","religion"," versus Not Cyberbullying")
    cm = confusion_matrix(df['target'], df['y_pred'])
    print("the type of the confusion matrix is ", type(cm))
    # Print the confusion matrix
    print(cm)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.show()
    
    cm_sum =  cm.sum()
    print("sum of numbers in cm is ", cm_sum)
    
    df['false_pos'] = np.where(df['target']==0, 1, 0) & np.where(df['y_pred']==1, 1, 0)
    df['false_neg'] = np.where(df['target']==1, 1, 0) & np.where(df['y_pred']==0, 1, 0)
    df_cnt_fp = df.groupby('label')['false_pos'].apply(lambda x: (x==True).sum()).reset_index(name='count')
    df_cnt_fn = df.groupby('label')['false_neg'].apply(lambda x: (x==True).sum()).reset_index(name='count')
    
    print('false positives')
    print(df_cnt_fp)
    print(type(df_cnt_fp))
    print(df_cnt_fp.axes)
    fp = df_cnt_fp.loc[df_cnt_fp['label']=='Religion', 'count'].values[0]
    fn = df_cnt_fn.loc[df_cnt_fn['label']=='Religion', 'count'].values[0]
    print(type(fp))
    print(fp)
    
    print('false negatives')
    print(df_cnt_fn)

    # Create each sub-confusion matrix of 2 x 2 for the five traits
    # Test hard coded for religion
    total_religion = count.get('Religion')  # 1575
    print("total in religion is ", total_religion)

    total_true_religion = cm[0][0]
    total_true_notcb = cm[1][1]
    # total_false_religion = df_cnt_fp.get(5)  #61
    # total_false_notcb_in_religion_model = df_cnt_fn.get('false_neg') # 0


    cm_religion = np.array([[total_true_religion, fp], [fn, total_true_notcb]])
    
    print(cm_religion)

    



if __name__=="__main__":
    parse_tvn()
    #graph_by_trt(df, cm)
