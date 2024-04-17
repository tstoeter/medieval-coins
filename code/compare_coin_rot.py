import os
import glob
import numpy as np
import sys
import scipy as sp
from global_config import *
from detect_coin import *
from analyse_detections import normalized_circle
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as ocv2


def get_roi(img1, c1, img2, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
     
    u1 = np.max(img1.shape)
    h1,w1,_ = img1.shape
    u2 = np.max(img2.shape)
    h2,w2,_ = img2.shape
    
    x1 = int(x1*u1)
    y1 = int(y1*u1)
    r1 = int(r1*u1)
    
    x2 = int(x2*u2)
    y2 = int(y2*u2)
    r2 = int(r2*u1)

    r = np.max([r1,r2])
    w = np.max([w1+r,w2+r,2*(x1+r),2*(x2+r)])
    h = np.max([h1+r,h2+r,2*(y1+r),2*(y2+r)])
    
    cx = w//2
    cy = h//2
    
    canvas1 = np.zeros([h,w,3])
    canvas2 = np.zeros([h,w,3])
    
    canvas1[cy-y1:cy-y1+h1 , cx-x1:cx-x1+w1] = img1
    canvas2[cy-y2:cy-y2+h2 , cx-x2:cx-x2+w2] = img2
    
    r = np.min([r1*u1,r2*u2])
    
    return canvas1, canvas2, r, r1


def circle_mask(img, r):
    h,w,_ = img.shape
    mask = np.zeros((h,w), dtype=np.uint8)
    ocv2.circle(mask, (h//2,w//2), r, 255, -1)
    
    return (mask > 0)


def cosine_sim(img1, img2, r1):
    
    if img1.size == 0 or img2.size == 0:
        return 0, 0, np.NaN
      
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
    
      
    vdot = np.vectorize(np.dot, signature="(3),(3)->()")
    result = vdot(img1[:,:], img2[:,:])
      
    mask1 = norm1 > 0.1
    mask2 = norm2 > 0.1
    mask3 = np.ones_like(mask2) > 0
    if (r1 > 0):
        mask3 = circle_mask(img1, r1)
    mask = (mask1 & mask2) & mask3
    
    inter = np.sum((mask1 & mask2)[:,:])
    union = np.sum((mask1 | mask2)[:,:])
    
    return result.reshape(img1.shape[0:2]), inter/(union+inter), np.median(result[mask].flatten())


def zcorr_sim(img1, img2, r1):
      
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
    
    mask1 = norm1 > 0.1
    mask2 = norm2 > 0.1
    mask3 = np.ones_like(mask2) > 0
    if (r1 > 0):
        mask3 = circle_mask(img1, r1)
    mask = (mask1 & mask2) & mask3
    
    inter = np.sum((mask1 & mask2)[:,:])
    union = np.sum((mask1 | mask2)[:,:])
    
    z1 = (img1[mask])[:,2].flatten()
    z2 = (img2[mask])[:,2].flatten()
    
    zcorr = np.corrcoef(z1, z2)[0,1]
    
    return img1, inter/(union+inter), zcorr


def diff_sim(img1, img2, r1):
    
    diff = img1-img2      
    norm = np.linalg.norm(diff, axis=-1)
    sim = 1.0/(1.0+norm)
    
    norm1 = np.linalg.norm(img1, axis=-1)
    norm2 = np.linalg.norm(img2, axis=-1)
      
    mask1 = norm1 > 0.1
    mask2 = norm2 > 0.1
    mask3 = np.ones_like(mask2) > 0
    if (r1 > 0):
        mask3 = circle_mask(img1, r1)
    mask = (mask1 & mask2) & mask3
    
    inter = np.sum((mask1 & mask2)[:,:])
    union = np.sum((mask1 | mask2)[:,:])
    
    return diff, inter/(union+inter), np.median(sim[mask].flatten())


metrics = {
    'diff': diff_sim,
    'cos': cosine_sim,
    'zcorr': zcorr_sim,
    }

vmin = {'diff': 0.6, 'cos': 0.9, 'zcorr': -0.5}
vmax = {'diff': 0.9, 'cos': 1.0, 'zcorr': 0.5}


def local_sim(cv1,cv2,sim,r1):
    N = 20
    h,w,_ = cv1.shape    
    img = np.zeros((h,w))
    
    for y in range(h):
        for x in range(w):
            _,_,s = sim(cv1[y-N//2:y+N//2, x-N//2:x+N//2], cv2[y-N//2:y+N//2, x-N//2:x+N//2], 0)
            img[y,x] = s
            
    return img


def image_rotate(img, deg):
    # vectorized matmul
    vrot = np.vectorize(np.matmul, signature='(m,n,k),(k,k)->(m,n,k)')

    if deg % 360 == 0:
        return img
    
    theta = np.radians(-deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0,0,1)))
    return sp.ndimage.rotate(vrot(img,R), deg, reshape=False, order=1)


def sim_max_rot(cv1, cv2, sim, r1):
    
    def f(x):
        img = image_rotate(cv2, x)
        return sim(cv1, img, r1)
    
    def f2(x):
        _, _, m = f(x)
        return m     

    degs = np.linspace(0,359,360)
    vals = [f2(x) for x in degs]
    vals = np.array(vals)
    
    j = np.argmax(vals)
    return vals[j], degs[j], image_rotate(cv2, degs[j]), degs, vals


filename = sys.argv[1]

print("Computing global similarity for " + filename)

inp_path = outp_path + "/correct_bending/"

outp_path = outp_path + "/global_sim_rot/"
if not os.path.exists(outp_path):
    os.makedirs(outp_path)
outf = outp_path + os.path.basename(filename).removesuffix(".npy")

output = []
i = accum_idx # from global config

coin = np.load(filename)
coinmeta = coin_metadata(filename, data_path)
    
_,_,_,c = detect_coin(coin)
x2, y2, r2 = c[1][i], c[2][i], c[3][i]
u2 = np.max(coin.shape)


for c in coin_templates:
    flist = glob.glob(inp_path + c + "*corr_bending.npy")
    fname = flist[0]
    temp = np.load(fname)
    tempmeta = coin_metadata(fname, data_path)
    
    i = accum_idx # from global config
    _,_,_,c = detect_coin(temp)
    x1, y1, r1 = c[1][i], c[2][i], c[3][i]
    u1 = np.max(temp.shape)
                
    cv1, cv2, rmin, rr = get_roi(temp, (x1,y1,r1), coin, (x2,y2,r2))
    
    (h,w,_) = cv1.shape
    
    df = {}
    
    df["Template_Type"] = tempmeta['type']
    df["Template_ID"] = tempmeta['identifier']
    df["Template_File"] = fname
    df["Coin_Type"] = coinmeta['type']
    df["Coin_ID"] = tempmeta['identifier']
    df["Coin_File"] = filename
    
    for sim in metrics:
        m, r, rimg, degs, vals = bisection_sim(cv1, cv2, metrics[sim], rr)
        df["Sim_" + sim] = m
        
        locsimg = local_sim(cv1,rimg,metrics[sim],rr)
        
        plt.subplots(1,6,figsize=(11,2), layout="constrained")
        
        plt.subplot(1,6,1)
        plt.imshow(uvw2rgb(cv1))
        plt.axis("off")
        ax = plt.gca()
        circle1 = plt.Circle( (w//2,h//2), r1*u1, fill = False, color='m', lw=2)
        ax.set_aspect( 1 )
        ax.add_artist( circle1 )
        
        plt.subplot(1,6,2)
        plt.imshow(uvw2rgb(cv2))
        plt.axis("off")
        ax = plt.gca()
        circle2 = plt.Circle( (w//2,h//2), r2*u2, fill = False, color='y', lw=2)
        ax.set_aspect( 1 )
        ax.add_artist( circle2 )
        
        plt.subplot(1,6,3)
        plt.plot(degs, vals)
        plt.axvline(x=r, color='r', ls=':')
        plt.axhline(y=m, color='g', ls=':')
        plt.xticks([0,90,180,270,360])
        plt.yticks([vals.min(), vals.mean(), vals.max()])
        plt.xlabel("Rotation angle")
        plt.ylabel("Similarity")
        plt.subplot(1,6,4)
        plt.imshow(uvw2rgb(rimg))
        plt.axis("off")
        
        plt.subplot(1,6,5)
        plt.imshow(uvw2rgb(cv1 + rimg))
        ax = plt.gca()
        ax.set_aspect( 1 )
        circle1 = plt.Circle( (w//2,h//2), r1*u1, fill = False, color='m', lw=2)
        circle2 = plt.Circle( (w//2,h//2), r2*u2, fill = False, color='y', lw=2)
        ax.add_artist( circle1 )
        ax.add_artist( circle2 )
        plt.axis("off")
        
        plt.subplot(1,6,6)
        plt.imshow(locsimg, vmin=vmin[sim], vmax=vmax[sim])
        plt.colorbar()
        plt.axis("off")

        plt.savefig(outf + "_" + tempmeta['type'] + "_" + sim + ".pdf", dpi=300)
        
    output.append(df)

df = pd.DataFrame(output)
df.to_csv(outf+"_global-sim.csv", index=False)
