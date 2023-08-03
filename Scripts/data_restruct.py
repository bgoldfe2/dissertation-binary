# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import os

#os.chdir("/home/bruce/dev/dissertation-binary/Scripts/")
os.chdir("/content/drive/MyDrive/Github/dissertation-binary/Scripts/")
def parse_all_traits():

    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    num_trt = 6

    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        inner = []
        
        # Filter for each trait
        for trt in range(num_trt):
            six_trait=six_df.loc[six_df['target'] == trt]
            inner.append(six_trait)
        rtn.append(inner)
    
    return rtn

            
def parse_by_trait(tgt):

    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    
    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        print('file ', flnm, ' is shape ', six_df.shape)
        print(six_df.head())
        
        six_trait=six_df.loc[six_df['target'] == tgt]
        print('    trait ', tgt, 'is shape ',six_trait.shape)
        print(six_trait.head())
        print()
        rtn.append(six_trait)

    return rtn[0], rtn[1], rtn[2]

def make_trait_based_files():
    fld_path = '../Dataset/SixClass/'
    six = ['test.csv', 'train.csv', 'valid.csv']
    
    rtn = []

    # Iterate over three files for all six traits per file
    for flnm in six:
        # Read in the file
        six_df = pd.read_csv(''.join([fld_path,flnm]))
        #print('file ', flnm, ' is shape ', six_df.shape)
        #print(six_df.head())
        
        six_trait=six_df.loc[six_df['target'] == tgt]
        #print('    trait ', tgt, 'is shape ',six_trait.shape)
        #print(six_trait.head())
        #print()
        rtn.append(six_trait)

    return rtn[0], rtn[1], rtn[2]


if __name__=="__main__":

    traits ={ "Age":0, "Ethnicity": 1, "Gender": 2, "Notcb":3, "Others":4, "Religion": 5}
    #val, train, test = parse_by_trait(4)
    #print("num train ",len(train), " val ", len(val), " test ", len(test))

    
    all = parse_all_traits()
    for i,df in enumerate(all):
        print('file ', i, ' is ', len(df), ' traits')
        for i,trt_num in enumerate(df):
            print('    trait ', i, ' is ', len(trt_num))

    #val = all[0]  #[0]
    #test = all[1] # [0]
    #train = all[2] #[0]
    
    # Save to Binary folder
    fld = ["val", "test", "train"]
    for i, folder in enumerate(fld):
      #os.mkdir(''.join(['../Dataset/Binary/',folder]))
      for trait in range(0,6):
        all[i][trait].to_csv(''.join(['../Dataset/Binary/',folder,'/',str(trait),'.csv']), index=False)


   
      
    
