from utils import *
from global_config import *
import glob
import numpy as np
import random
import cv2
import json
import pandas as pd
import os
import re
import sys


def hit(x,y,r, tx,ty,tr1,tr2):
    d = np.array([x,y]) - np.array([tx,ty])
    if (r < tr1 or r > tr2):
        return 0
    if np.linalg.norm(d) > tr2-tr1:
        return 0
    return 1


def hit2(x,y,r, tx,ty,tr1,tr2):
    d = np.array([x,y]) - np.array([tx,ty])
    if (r < tr1 or r > tr2):
        return 0
    if r-np.linalg.norm(d) < tr1:
        return 0
    if np.linalg.norm(d)+r > tr2:
        return 0
    return 1


def errors(x,y,r, tx,ty,tr1,tr2):
    d = np.array([x,y]) - np.array([tx,ty])
    return [np.linalg.norm(d), r-tr1, tr2-r]


def normalized_circle(filename):

    with open(filename) as jf:
        jsondata = json.load(jf)
        
    x = jsondata['x']
    y = jsondata['y']
    r1 = jsondata['r1']
    r2 = jsondata['r2']
    
    cid = coinid_from_filename(filename)
    
    normalfile = os.path.basename(filename).removesuffix("_circle.json") + "*_corr_bending.npy"
    normalfiles = glob.glob(outp_path + "correct_bending/" + normalfile)
    normals = np.load(normalfiles[0])
    h,w,_ = normals.shape

    unit = np.max(normals.shape)
    
    return [cid,x,y,r1,r2,r1*unit,r2*unit, (r2-r1)*unit]


if __name__ == "__main__":
    filelist = glob.glob(rslt_path + "labels/*_circle.json")

    l = []
    
    for f in filelist:
        n = normalized_circle(f)
        l.append(n)
    
    df = pd.DataFrame(l, columns=["CoinID", "Target X", "Target Y", "Inner Radius", "Outer Radius", "Inner Radius mm", "Outer Radius mm", "Ring Width mm"])
    
    filelist = glob.glob(outp_path + "detected_coins/*detections.csv")
    
    df2 = pd.DataFrame()
    
    nacc = int(sys.argv[1]) # accumulator index 0..9
    nqnt = int(sys.argv[2]) # quantile index
    
    assert(nacc < 10)
    assert(nqnt < 50)
    
    for csvfile in filelist:
        d = pd.read_csv(csvfile)
        d = d.loc[[nqnt*10 + nacc]]
        del d['Unnamed: 0']
        df2 = pd.concat([df2, d])
    
    result = pd.merge(df, df2, how="inner", on="CoinID")
    
    for index, row in result.iterrows():
        h = hit(row['Center X'], row['Center Y'], row['Radius'], row['Target X'], row['Target Y'], row['Inner Radius'], row['Outer Radius'])
        h2 = hit2(row['Center X'], row['Center Y'], row['Radius'], row['Target X'], row['Target Y'], row['Inner Radius'], row['Outer Radius'])
        e = errors(row['Center X'], row['Center Y'], row['Radius'], row['Target X'], row['Target Y'], row['Inner Radius'], row['Outer Radius'])
        result.at[index, 'Hit'] = h
        result.at[index, 'Hit 2'] = h2
        result.at[index, 'Dist. Centers'] = e[0]
        result.at[index, 'Dist. Inner'] = e[1]
        result.at[index, 'Dist. Outer'] = e[2]
    
    print(nacc, nqnt, result["Quantile"][0], result["Accumulator"].mean(), result["Hit"].mean(), result["Hit 2"].mean(), result["Dist. Centers"].mean(), result["Dist. Inner"].mean(), result["Dist. Outer"].mean())
    