import os
import sys
import glob
import pandas as pd
import numpy as np

from global_config import *
from sklearn.metrics import *
from matplotlib import pyplot as plt


inp_path = outp_path + "/global_sim_rot/"

filelist = glob.glob(inp_path + "/*.csv")

d = inp_path.split('_')

dists = ["Sim_diff", "Sim_zcorr", "Sim_cos"]

for d in dists:
    
    df = pd.DataFrame()

    target = []
    predict = []
    
    for f in filelist:
        df2 = pd.read_csv(f)
        mind = df2[d].argmax()  
        target.append(df2.iloc[0]["Coin_Type"])
        predict.append(df2.iloc[mind]["Template_Type"])
        
        df = pd.concat([df, df2])

    print(df.describe())
    
    labels = ["Bahrfeldt 19", "Mehl 499", "Mehl 595"]
    labels2 = ["Bahrfeldt-19", "Mehl-499", "Mehl-595"]
    
    plt.figure()
    plt.rcParams.update({'font.size': 15})
    disp = ConfusionMatrixDisplay.from_predictions(target, predict, normalize='true', display_labels=labels)
    disp.im_.set_clim(0, 1)
    plt.rcParams.update({'font.size': 15})
    plt.savefig(d + "_confusion-matrix.pdf")
    
    mcc = matthews_corrcoef(target, predict)
    print("MCC ", d, "=", mcc)
    
    plt.figure(layout="constrained")
    
    mind = df[d].min()
    maxd = df[d].max()
    
    boxdata = []
    
    # pair-wise different coins
    sel = df.loc[(df["Template_Type"] != df["Coin_Type"])]
    plt.subplot(4,1,1)
    plt.hist(sel[d], 20, range=[mind, maxd])
    
    boxdata.append(sel[d])
    
    for i,l in enumerate(labels2):
        sel = df.loc[(df["Template_Type"] == l) & (df["Coin_Type"] == l)]
        plt.subplot(4,1,2+i)
        plt.hist(sel[d], 20, range=[mind, maxd])
        boxdata.append(sel[d])
    
    plt.savefig(d + "_histogram.pdf")

    plt.figure(layout="constrained")
    boxdata.reverse()
    plt.boxplot(boxdata, vert=False, labels=["Mehl 595","Mehl 499","Bahrfeldt 19","All other coins"])
    plt.savefig(d + "_boxplot.pdf")
    
    plt.figure()
    xticks = []
    xlabels = []
    
    labels = ["Bahrfeldt 19","Mehl 499","Mehl 595"]
        
    # grouped boxplot
    for i,l in enumerate(labels2):
        boxdata = []
        for j,m in enumerate(labels2):
            sel = df.loc[(df["Template_Type"] == m) & (df["Coin_Type"] == l)]
            print(m,l, ":", sel[d].describe())
            boxdata.append(sel[d])
            
        bp = plt.boxplot(boxdata, patch_artist=True, widths=0.8, positions=np.arange(1+4*i, 4+4*i), labels=labels)
        colors = ['pink', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        xlabels.append(l)
        xticks.append(4*i+2)

    for label, color in zip(labels,colors):
        plt.plot([], c=color, label=label)
        plt.legend()
    
    plt.gca().yaxis.grid(True)
    plt.xticks(xticks)
    plt.xlabel("Reference coins")
    plt.ylabel("Similarity")
    plt.savefig(d + "_groupbox.pdf")
      
    
    mcc = matthews_corrcoef(target, predict)
    print("MCC ", d, "=", mcc)
    print(classification_report(target, predict, target_names=labels))
    