import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import json

from global_config import *
from utils import *
from detect_coin import detect_coin
from analyse_detections import normalized_circle, hit, hit2


filelist = glob.glob(rslt_path + "labels/*_circle.json")

plt.subplots(11, 10, figsize=(10, 12))

v = accum_idx   # nacc, accumulator index
num = 0 # number of hits

for i,f in enumerate(filelist):
    print(f)
    n = normalized_circle(f)
    normalfile = os.path.basename(f).removesuffix("_circle.json") + "*_corr_bending.npy"
    print(normalfile)
    normalfiles = glob.glob(outp_path + "correct_bending/" + normalfile)
    print(normalfiles[0])
    normals = np.load(normalfiles[0])

    plt.subplot(11,10,i+1)
    plt.imshow(uvw2rgb(normals))
    unit = np.max(normals.shape)
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    tx, ty = n[1]*unit, n[2]*unit
    tr1 = n[3]*unit
    tr2 = n[4]*unit
    
    circle = plt.Circle( (tx,ty), tr1, fill = False, color="m")
    ax.set_aspect( 1 )
    ax.add_artist( circle )

    circle = plt.Circle( (tx,ty ), tr2, fill = False, color="m")
    ax.set_aspect( 1 )
    ax.add_artist( circle )
    
    _,_,_,c = detect_coin(normals)
    
    for v in range(5):
    
        x, y = c[1][v]*unit, c[2][v]*unit
        r = c[3][v]*unit
            
        circle = plt.Circle( (x,y), r, fill = False, color='y')
        ax.set_aspect( 1 )
        ax.add_artist( circle )
    
    if hit(x,y,r, tx,ty,tr1,tr2) == 0:
        # ax.set_title("miss")
        ax.patch.set_linewidth(10)
        ax.patch.set_edgecolor('r')
    else:
        num += 1

plt.subplot(11,10, 110)
ax = plt.gca()
ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig("detections_labels.pdf")


plt.subplots(1,3)

for i,f in enumerate(coin_templates):
    print(f)
    normalfile = "*" + f + "*_corr_bending.npy"
    print(normalfile)
    normalfiles = glob.glob(outp_path + "correct_bending/" + normalfile)
    print(normalfiles[0])
    normals = np.load(normalfiles[0])

    plt.subplot(1,3,i+1)
    plt.imshow(uvw2rgb(normals))
    unit = np.max(normals.shape)
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    _,_,_,c = detect_coin(normals)
      
    x, y = c[1][v]*unit, c[2][v]*unit
    r = unit*0.31
        
    circle = plt.Circle( (x,y), r, fill = False, color='m', lw=3)
    ax.set_aspect( 1 )
    ax.add_artist( circle )
    
    x, y = c[1][v]*unit, c[2][v]*unit
    r = unit*0.41
        
    circle = plt.Circle( (x,y), r, fill = False, color='m', lw=3)
    ax.set_aspect( 1 )
    ax.add_artist( circle )
    
    x, y = c[1][v]*unit, c[2][v]*unit
    r = c[3][v]*unit
        
    circle = plt.Circle( (x,y), r, fill = False, color='y', lw=3)
    ax.set_aspect( 1 )
    ax.add_artist( circle )
    
plt.tight_layout()

plt.savefig("detections_templates.pdf")
