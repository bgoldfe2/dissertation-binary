# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023
# adapted from prior work

import matplotlib.pyplot as plt
from collections import defaultdict
from Model_Config import traits


def save_acc_curves(args, trt, history):
    #print(history)
    #trt = 0
    # set the y-axis limits
    
    plt.plot(range(1,5),history['train_acc'], label='train accuracy')
    plt.plot(range(1,5),history['val_acc'], label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xlim(1, 4)
    plt.xticks(range(1, 5))
    plt.ylim([0.7, 1.0])
    plt.savefig(f"{args.figure_path}{traits.get(str(trt))}---acc---.pdf")
    #plt.clf()
    return plt

def save_loss_curves(args, trt, history):
    #print(history)
    #trt=0
    plt.plot(range(1,5),history['train_loss'], label='train loss')
    plt.plot(range(1,5),history['val_loss'], label='validation loss')
    plt.title('Training and Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xlim(1, 4)
    plt.xticks(range(1, 5))
    plt.ylim([0.0, 0.3])
    plt.savefig(f"{args.figure_path}{traits.get(str(trt))}---loss---.pdf")
    #plt.clf()
    return plt

# This version adds the plt.clf() command at the end


if __name__=="__main__":
    history = defaultdict(list)
    history['train_acc'] = [0.8809, 0.9798, 0.9864, 0.991]
    history['val_acc'] = [0.9715, 0.9854, 0.9864, 0.9857]
    history['train_loss'] = [0.2564959865777443, 0.08274737073108554, 0.05754305351215104, 0.04171089528749387]
    history['val_loss'] = [0.11794130202157027, 0.057937183756042614, 0.055730998901782014, 0.058969416937819034]
    
    plot = save_acc_curves( history)
   
    plot2 = save_loss_curves( history)

    