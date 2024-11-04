import os
import sys
import cv2
import numpy as np
import scipy as sp
from global_config import *
from utils import *

filename = sys.argv[1]

outp_path = outp_path + "/correct_bending/"
if not os.path.exists(outp_path):
    os.makedirs(outp_path, exist_ok=True)
outf = outp_path + os.path.basename(filename).removesuffix(".npy")

print("Correct bending for " + filename)

normals = np.load(filename)
mask = get_mask(normals)

w = 31
bending_model = np.zeros_like(normals)

bending_model[:,:,0] = cv2.GaussianBlur(normals[:,:,0], (w,w), 0)
bending_model[:,:,1] = cv2.GaussianBlur(normals[:,:,1], (w,w), 0)

normals_nobend = normalize(np.multiply((normals - bending_model), mask))

np.save(outf+"_bending.npy", bending_model)
np.save(outf+"_corr_bending.npy", normals_nobend)

cv2.imwrite(outf+"_bending.png", cv2.cvtColor(uvw2rgb(bending_model), cv2.COLOR_RGB2BGR))
cv2.imwrite(outf+"_corr_bending.png", cv2.cvtColor(uvw2rgb(normals_nobend), cv2.COLOR_RGB2BGR))
