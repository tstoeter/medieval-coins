import os
import sys
import numpy as np
import cv2

from global_config import *
from utils import *


target_ppmm = int(sys.argv[1])
filename = sys.argv[2]

outp_path = outp_path + "/coins_" + str(target_ppmm) + "ppmm/"
if not os.path.exists(outp_path):
    os.makedirs(outp_path)
outf = outp_path + os.path.basename(filename).removesuffix(".tif") + "_" + str(target_ppmm) + "ppmm.npy"

print("Downscaling " + filename + " to " + outf + " at " + str(target_ppmm))

coin = bgr2uvw(cv2.imread(filename))
meta = coin_metadata(filename, data_path)

h,w,_ = coin.shape
h = int(h * target_ppmm / meta["ppmm"])
w = int(w * target_ppmm / meta["ppmm"])

coin_res = cv2.resize(coin, (w,h), cv2.INTER_AREA)
coin_res = normalize(coin_res)

np.save(outf, coin_res)
