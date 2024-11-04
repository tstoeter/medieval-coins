import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.transform import hough_circle, hough_circle_peaks
from global_config import *
from utils import *


def make_radial_field(shape):
    h = shape[0]
    w = shape[1]

    out = np.zeros((h, w, 3))

    for y in range(h):
        for x in range(w):
           out[y, x] = np.array([(x-w/2), (-y+h/2), 0])
           l = np.linalg.norm(out[y,x])
           if l > 0:
                out[y,x] /= l

    return out


def vector_field_dot(vf1, vf2):
    assert(vf1.shape == vf2.shape)
    shape = vf1.shape
    vdot = np.vectorize(np.dot, signature="(3),(3)->()")
    return vdot(vf1[:, :], vf2[:, :]).reshape((shape[0], shape[1]))


def detect_coin(normals, qt=0.76):

    def __round_odd(x):
        x = int(x)
        if (x % 2) == 1:
            return x
        return x + 1
    
    shape = normals.shape

    radial_field = make_radial_field(shape)

    w = 21
    ring_amap = vector_field_dot(radial_field, normals)

    q = np.quantile(ring_amap[:, :], qt)
    _, thmap = cv2.threshold(ring_amap, q, 255, cv2.THRESH_TOZERO)
    
    # hough transform to detect circles
    w = np.max(shape)
    r1 = 0.31 * w
    r2 = 0.41 * w
    hough_radii = np.arange(int(r1), int(r2))
    hough_res = hough_circle(thmap, hough_radii)
    
    acc, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=10)

    return normals, ring_amap, thmap, (acc, cx/w, cy/w, radii/w)


if __name__ == "__main__":
 
    filename = sys.argv[1]
    print("Detecting coin " + filename)
    normals = np.load(filename)

    meta = coin_metadata(filename, data_path)

    outp_path = outp_path + "/detected_coins/"
    if not os.path.exists(outp_path):
        os.makedirs(outp_path, exist_ok=True)
    outf = outp_path + os.path.basename(filename).removesuffix(".npy") + "_detections" + ".csv"
    
    out_df = pd.DataFrame()

    
    if coin_templates[1] in filename:
        radial_field = make_radial_field(normals.shape)
    
        plt.figure()
        ax = plt.subplot(151)
        X = np.arange(-4, 5, 1)
        Y = np.arange(-4, 5, 1)
        U, V = np.meshgrid(X, Y)
        N = np.sqrt(U**2+V**2)  # there may be a faster numpy "normalize" function
        U2, V2 = U/N, V/N
        
        ax.quiver(X, Y, U2, V2)
        ax.set_aspect(1)
        plt.axis('off')

        normals_nobg, ring_amap, thmap, (acc, cx, cy, r) = detect_coin(normals)

        plt.subplot(152)
        plt.imshow(uvw2rgb(normals))
        plt.axis('off')
        plt.subplot(153)
        plt.imshow(ring_amap)
        plt.axis('off')
        plt.subplot(154)
        plt.imshow(thmap)
        plt.axis('off')

        plt.subplot(155)
        plt.imshow(uvw2rgb(normals))
        plt.axis('off')
        unit = np.max(normals.shape)
        ax = plt.gca()
        for i in [accum_idx]: #range(len(r)):
            circle = plt.Circle( (cx[i]*unit, cy[i]*unit ), r[i]*unit, fill = False, color="yellow", lw=2)
            ax.set_aspect( 1 )
            ax.add_artist( circle )
            
    plt.tight_layout()
    plt.savefig("coin_rim_steps.pdf")
    
    for qt in range(60,91):
        normals_nobg, ring_amap, thmap, (acc, cx, cy, r) = detect_coin(normals, qt/100)
        df = pd.DataFrame({"CoinJSON": meta["jsonFile"], "CoinID": meta["identifier"], "CoinType": meta["type"], "Quantile": qt, "Accumulator": acc, "Center X": cx, "Center Y": cy, "Radius": r})
        out_df = pd.concat([out_df, df])
    
    out_df.to_csv(outf)
    