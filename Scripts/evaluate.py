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

from engine import test_eval_fn
from Model_Config import Model_Config, traits

from utils import oneHot, roc_curve, auc

def test_evaluate(trt, test_df, test_data_loader, model, device, args: Model_Config):

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
    history2['Classification_Report'] = cls_rpt

    with open(f'{args.output_path}{traits.get(str(trt))}---test_metrics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in history2.items():
            writer.writerow([key, value])

    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(f'{args.output_path}{traits.get(str(trt))}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)
    # auc evaluation new for this version

    #ROC Curve

    calc_roc_auc(trt, np.array(y_test), np.array(y_proba), args)

    # Return the test results for saving in train.py
    return pred_test, acc

def calc_roc_auc(trt, all_labels, all_logits, args, name=None ):

    trait = traits.get(str(trt))
    notcb = traits.get(str(3))
    attributes = [trait, notcb ]
    #attributes = ['Age', 'Ethnicity', 'Gender', 'Notcb', 'Others', 'Religion']
    
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
        plt.savefig(f"{args.figure_path}{trait}---roc_auc_curve---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')