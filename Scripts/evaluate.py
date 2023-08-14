# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import pandas as pd
from visualize import make_confusion_matrix

from engine import test_eval_fn
from Model_Config import Model_Config, traits

from utils import oneHot, roc_curve, auc, generate_dataset_for_ensembling, load_models, set_device

def test_evaluate(trt, test_df, test_data_loader, model, device, args: Model_Config, *ens_flag):

    if ens_flag:
        print("called from ensemble")
    else:
        print("not called from ensemble")
        asdf

    history2 = defaultdict(list)

    # modified using the Model_Config instance args as the state reference
    pretrained_model = args.pretrained_model
    print(f'\nEvaluating: ---{pretrained_model}---\n')
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, model, device, args)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cls_rpt = classification_report(y_test, y_pred, digits=4)
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', cls_rpt)

    history2['Accuracy'] = acc
    history2['MCC'] = mcc
    history2['Precision'] = precision
    history2['Recall'] = recall
    history2['F1_score'] = f1
    history2['Classification_Report\\n'] = cls_rpt

    with open(f'{args.output_path}{traits.get(str(trt))}---test_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in history2.items():
            writer.writerow([key, value])

    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.output_path}{traits.get(str(trt))}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    history2['conf_mat'] = conf_mat
    print(conf_mat)
    plt.figure(3)

    # group_names = ['True Neg','False Pos','False Neg','True Pos']
    # group_counts = ['{0:0.0f}'.format(value) for value in conf_mat.flatten()]
    # group_percentages = ['{0:.2%}'.format(value) for value in
    #                     conf_mat.flatten()/np.sum(conf_mat)]
    # labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
    #         zip(group_names,group_counts,group_percentages)]
    # labels = np.asarray(labels).reshape(2,2)

    # conf_plt = sns.heatmap(conf_mat/np.sum(conf_mat), 
    #                        annot=labels, 
    #                        fmt='',
    #                        linecolor='white',
    #                        linewidths=1,
    #                        cmap='Blues')  

    labels = ['True Pos','False Pos','False Neg','True Neg']
    categories = ['1', '0']
    make_confusion_matrix(args, trt, conf_mat, 
                      group_names=labels,
                      categories=categories, 
                      cmap='Blues',
                      title=traits.get(str(trt)))
    
    #conf_plt.set(xlabel=trt, ylabel='Notcb')
    # plt.savefig(''.join([args.figure_path, traits.get(str(trt)), 'confusion_matrix.pdf']), dpi=400)
    # plt.clf()
    # plt.close()
    # auc evaluation new for this version
    # ROC Curve
    calc_roc_auc(trt, np.array(y_test), np.array(y_proba), args)

    # Return the test results for saving in train.py
    return pred_test, acc

def calc_roc_auc(trt, all_labels, all_logits, args, name=None ):

    trait = traits.get(str(trt))
    notcb = traits.get(str(3))
    attributes = [trait, notcb ]
    print("attributes in calc_roc_auc are ", attributes)
    
    
    all_labels = oneHot(all_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(2)
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
        plt.savefig(f"{args.figure_path}{trait}---roc_auc_curve---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')

def evaluate_all_models(args: Model_Config):
    all_trait_models = load_models(args)
    #print("&*&**&*&&*&*&*&*&*&* ",len(all_trait_models))

    # TODO combine all the traits into binary dataset with a mapping back to 
    # their original full dataset values

    #test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device(args)

    test_data_path = '../Dataset/Binary/test/'

    df = pd.read_csv(''.join([test_data_path, 'test_Age.csv']))
    print(df.head())

    # TODO Do I want the Notcb to be tested? Should there be a Notcb model which is an Notcb vs the rest?
    trt_pop = traits.pop('3', 'no key found')
    just_cb = traits.values()
    print('just cb is ',just_cb)
    #print("trt popped out is ", trt_pop)
    all_test_data = []
    for trt_cb in just_cb:
        all_test_data.append(pd.read_csv(''.join([test_data_path, 'test_', trt_cb, '.csv'])))

    test_df = pd.concat(all_test_data, axis=0).reset_index()

    
    # loop through all the models by trait in list of name: model dictionary in 
    # all_trait_models
    for trt, trt_mdl in all_trait_models.items():
        print("trait is ", trt)
        test_df.loc[test_df['label'] == trt, 'target'] = 0
        test_df.loc[test_df['label'] != trt, 'target'] = 1
        trt_mdl.to(device)
        args.pretrained_model="roberta-base"
        test_data_loader = generate_dataset_for_ensembling(args, df=test_df)
        print("********************* Evaluating Model for Trait", trt, " *************************")
        ensemble=True
        test_evaluate(trt, test_df, test_data_loader, trt_mdl, device, args, ensemble)
        del trt_mdl, test_data_loader