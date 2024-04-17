import os
import sys
import cv2
import numpy as np
import scipy as sp
import glob
from global_config import *
from utils import *
from matplotlib import pyplot as plt
import pandas as pd
from detect_coin import *


filename = sys.argv[1]

print("Computing histogram distances for " + filename)

inp_path = outp_path + "/correct_bending/"

outp_path = outp_path + "/histogram_dist/"
if not os.path.exists(outp_path):
    os.makedirs(outp_path)
outf = outp_path + os.path.basename(filename).removesuffix(".npy")

def transform_coordinates(c, x, u):
    # build local coordinate system
    x[0] = c[0]-x[0]
    x[1] = c[1]-x[1]
    x = x/np.linalg.norm(x)
    y = np.flip(x)
    y[0] = -y[0]

    radial = np.dot(u,x)
    ortho = np.dot(u,y)
    
    return np.array([radial, ortho])

def get_hist(data, x, y, r):
    
    c = np.array([x, y])

    # get data as list of valid normals
    h,w,d = np.shape(data)
    data2 = data.reshape(w*h,d)
    mask2 = get_mask(data).reshape(w*h,d)
    mask3 = np.equal(data2, mask2)
    mask4 = np.logical_and(np.logical_and(mask3[:,0], mask3[:,1]), mask3[:,2])
    mask5 = np.logical_not(mask4)
    data3 = data2[mask5]
    
    # histogram over grad field
    grad_field = data3[:,0:2] / data3[:,2:3]
    
    datalist = []
    
    for j in range(h):
        for i in range(w):
            if np.linalg.norm(data[j,i]) <= 0.1:
                continue
            p = np.array([i, j])
            u = data[j,i] / data[j,i,2]
            h = transform_coordinates(c, p, u[0:2])
            datalist.append(h)
    
    datalist = np.array(datalist)
    
    hist2d,xbins,ybins = np.histogram2d(datalist[:,0], datalist[:,1], bins=25, range=[[-0.5, 0.5], [-0.5, 0.5]], density=True)
      
    return hist2d

def hist_sims(h1, h2):
    emd,_,flow = cv2.EMD(h1.astype(np.float32), h2.astype(np.float32), cv2.DIST_L2)
    chi = 0.5*cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CHISQR_ALT)
    return emd,chi


# load tempalte coins

coin = np.load(filename)
coinmeta = coin_metadata(filename, data_path)
h,w,_ = coin.shape

coinhist = get_hist(coin, w//2, h//2, 0)
np.save(outf+"_grad-hist.npy", coinhist)


plt.figure()
plt.rcParams.update({'font.size': 20})
ax = plt.gca()
plt.imshow(coinhist, vmin=0, vmax=10)
ax.set_xticks([0,12,24], labels=["-0.5", "0.0", "0.5"])
ax.set_yticks([0,12,24], labels=["-0.5", "0.0", "0.5"])
plt.colorbar()
plt.rcParams.update({'font.size': 20})
plt.xlabel("Radial")
plt.ylabel("Circular")
plt.savefig(outf+"_grad-hist.pdf")

output = []

for c in coin_templates:
    flist = glob.glob(inp_path + c + "*corr_bending.npy")
    fname = flist[0]
    temp = np.load(fname)
    tempmeta = coin_metadata(fname, data_path)
    h,w,_ = temp.shape
    temphist = get_hist(temp, w//2, h//2, 0)

    emd,chi = hist_sims(temphist, coinhist)
    
    df = {}
    
    df["Template_Type"] = tempmeta['type']
    df["Template_ID"] = tempmeta['identifier']
    df["Template_File"] = fname
    df["Coin_Type"] = coinmeta['type']
    df["Coin_ID"] = tempmeta['identifier']
    df["Coin_File"] = filename
    df["Dist_EMD"] = emd
    df["Dist_Chi2"] = chi
    
    output.append(df)

df = pd.DataFrame(output)
df.to_csv(outf+"_hist-sim.csv", index=False)
