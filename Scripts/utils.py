import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from model import DeBertaFGBC, RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC, GPT_Neo13FGBC
from dataset import DatasetDeberta, DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo, DatasetGPT_Neo13

import os
from datetime import datetime
from Model_Config import Model_Config

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_folders(args: Model_Config) -> Model_Config:
    # Create the Runs Experiment folder location for Model,s Output, Figures
    # Get current time, remove microseconds, replace spaces with underscores
    current_datetime = str(datetime.now().replace(microsecond=0)).replace(" ","_")
    
    # NOTE: for multi-architecture runs this run will append only the first model type
    folder_name = "../Runs/" + current_datetime.replace(':','_') + "--" + args.pretrained_model.split('/',1)[1]
    n=7 # number of letters in Scripts which is the folder we should be running from
    cur_dir = os.getcwd()
    #print(cur_dir)
    #print('folder name ', folder_name)
    
    # Parse out any subfolders for model descriptors e.g. microsoft/DeBERTa
    foo = args.model_list
    subfolders = []
    for bar in foo:
        if '/' in bar:
            fubar = bar.split('/',1)[0]
            subfolders.append(fubar)
    print(subfolders)

    # High level folders defined
    fld = ['/Models/', '/Output/', '/Figures/']
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    print('args.model_path are\n',args.model_path)
    
    if cur_dir[len(cur_dir)-n:] != 'Scripts':
        print('Run driver.py from Scripts Directory')        
    else:
        # Make the parent folder for this run
        os.mkdir(folder_name)

        # Create the subfolders as needed for models
        top_list = []
        for top in fld:
            fld_name = folder_name + top
            print(fld_name)
            top_list.append(fld_name)
            os.mkdir(fld_name)
        for sub in subfolders:
            for top in top_list:
                sub_name = top + sub + '/'
                print(sub_name)
                os.mkdir(sub_name)

    print('args type ', type(args))
    print('args.model path value ', args.model_path)

    return args

def set_device(args):
    device = ""
    if(args.device=="cpu"):
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(device=="cpu"):
            print("GPU not available.")
    return device

def sorting_function(val):
    return val[1]    

def load_prediction(args):
    deberta_path = (f'{args.output_path}microsoft/deberta_v3_base.csv')
    xlnet_path = (f'{args.output_path}xlnet-base-cased.csv')
    roberta_path = (f'{args.output_path}roberta-base.csv')
    albert_path = (f'{args.output_path}albert-base-v2.csv')
    gpt_neo_path = (f'{args.output_path}EleutherAI/gpt-neo-125M.csv')
    gpt_neo_path = (f'{args.output_path}EleutherAI/gpt-neo-1.3B.csv')

    deberta = pd.read_csv(deberta_path)
    xlnet = pd.read_csv(xlnet_path)
    roberta = pd.read_csv(roberta_path)
    albert = pd.read_csv(albert_path)
    gpt_neo = pd.read_csv(gpt_neo_path)

    return deberta, xlnet, roberta, albert, gpt_neo

def print_stats(max_vote_df, deberta, xlnet, roberta, albert):
    print(max_vote_df.head())
    print(f'---Ground Truth---\n{deberta.target.value_counts()}')
    print(f'---DeBerta---\n{deberta.y_pred.value_counts()}')
    print(f'---XLNet---\n{xlnet.y_pred.value_counts()}')
    print(f'---Roberta---\n{roberta.y_pred.value_counts()}')
    print(f'---albert---\n{albert.y_pred.value_counts()}')

def evaluate_ensemble(max_vote_df, args):
    y_test = max_vote_df['target'].values
    y_pred = max_vote_df['pred'].values
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', classification_report(y_test, y_pred, digits=4))
    
    max_vote_df.to_csv(f'{args.output_path}Ensemble-{args.ensemble_type}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)

def generate_dataset_for_ensembling(pretrained_model, df, args):
    if(pretrained_model == "microsoft/deberta-v3-base"):
        dataset = DatasetDeberta(text=df.text.values, target=df.target.values, pretrained_model="microsoft/deberta-v3-base")
    elif(pretrained_model== "roberta-base"):
        dataset = DatasetRoberta(text=df.text.values, target=df.target.values, pretrained_model="roberta-base")
    elif(pretrained_model== "xlnet-base-cased"):
        dataset = DatasetXLNet(text=df.text.values, target=df.target.values, pretrained_model="xlnet-base-cased")
    elif(pretrained_model == "albert-base-v2"):
        dataset = DatasetAlbert(text=df.text.values, target=df.target.values, pretrained_model="albert-base-v2")
    elif(pretrained_model == "EleutherAI/gpt-neo-125M"):
        dataset = DatasetGPT_Neo(text=df.text.values, target=df.target.values, pretrained_model="EleutherAI/gpt-neo-125M")
    elif(pretrained_model == "EleutherAI/gpt-neo-1.3M"):
        dataset = DatasetGPT_Neo13(text=df.text.values, target=df.target.values, pretrained_model="EleutherAI/gpt-neo-1.3B")

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    return data_loader

def load_models(args):
    deberta_path = (f'{args.model_path}microsoft/deberta-v3-base_Best_Val_Acc.bin')
    xlnet_path = (f'{args.model_path}xlnet-base-cased_Best_Val_Acc.bin')
    roberta_path = (f'{args.model_path}roberta-base_Best_Val_Acc.bin')
    albert_path = (f'{args.model_path}albert-base-v2_Best_Val_Acc.bin')
    gpt_neo_path = (f'{args.model_path}EleutherAI/gpt-neo-125M_Best_Val_Acc.bin')
    gpt_neo13_path = (f'{args.model_path}EleutherAI/gpt-neo-1.3B_Best_Val_Acc.bin')

    #deberta = DeBertaFGBC(pretrained_model="microsoft/deberta-v3-base")
    #xlnet = XLNetFGBC(pretrained_model="xlnet-base-cased")
    #roberta = RobertaFGBC(pretrained_model="roberta-base")
    #albert = AlbertFGBC(pretrained_model="albert-base-v2")
    #gpt_neo = GPT_NeoFGBC(pretrained_model="EleutherAI/gpt-neo-125M")
    #gpt_neo13 = GPT_Neo13FGBC(pretrained_model="EleutherAI/gpt-neo-1.3B")

    # Need to change the args.pretrained_model first then send args as parameter?
    # Similar to train.py set_model() function line 214
    
    deberta = DeBertaFGBC(pretrained_model="microsoft/deberta-v3-base")
    xlnet = XLNetFGBC(pretrained_model="xlnet-base-cased")
    roberta = RobertaFGBC(pretrained_model="roberta-base")
    albert = AlbertFGBC(pretrained_model="albert-base-v2")
    gpt_neo = GPT_NeoFGBC(pretrained_model="EleutherAI/gpt-neo-125M")
    gpt_neo13 = GPT_Neo13FGBC(pretrained_model="EleutherAI/gpt-neo-1.3B")

    deberta.load_state_dict(torch.load(deberta_path))
    xlnet.load_state_dict(torch.load(xlnet_path))
    roberta.load_state_dict(torch.load(roberta_path))
    albert.load_state_dict(torch.load(albert_path))
    gpt_neo.load_state_dict(torch.load(gpt_neo_path))
    gpt_neo13.load_state_dict(torch.load(gpt_neo13_path))

    return deberta, xlnet, roberta, albert, gpt_neo, gpt_neo13

def oneHot(arr):
    b = np.zeros((arr.size, arr.max()+1))
    b[np.arange(arr.size),arr] = 1
    return b

def calc_roc_auc(all_labels, all_logits, args, name=None, ):
    attributes = []
    if(args.classes==6):
       attributes = ['Age', 'Ethnicity', 'Gender', 'Notcb', 'Others', 'Religion']
    elif(args.classes==5):
        #attributes = ['Age', 'Ethnicity', 'Gender', 'Religion', 'Others',]
        attributes = ['Age', 'Ethnicity', 'Gender', 'Religion', 'Others']

    elif(args.classes==3):
        attributes = ['None', 'Sexism', 'Racism']
    all_labels = oneHot(all_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0,len(attributes)):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s %g' % (attributes[i], roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    if (name!=None):
        plt.savefig(f"{args.figure_path}{name}---roc_auc_curve---.pdf")
    else:
        plt.savefig(f"{args.figure_path}{args.pretrained_model}---roc_auc_curve---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')