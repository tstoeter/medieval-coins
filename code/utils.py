import os
import json
import numpy as np
import glob
import re
from global_config import *


def __bgr2uvw(rgb):
    if rgb[0] == rgb[1] == rgb[2] == 127:
        #return np.array([np.nan, np.nan, np.nan])
        return np.array([0,0,0])
    out = np.zeros_like(rgb).astype(np.float64)
    out[0] = -1.0 + (2*rgb[2] / 255)
    out[1] = -1.0 + (2*rgb[1] / 255)
    out[2] = -1.0 + (2*rgb[0] / 255)
    l = np.linalg.norm(out)
    return out/l


def bgr2uvw(img):
    h,w,_ = img.shape
    out = np.zeros_like(img).astype(np.float64)
    for y in range(h):
        for x in range(w):
            out[y,x] = __bgr2uvw(img[y,x])
    return out

def __rgb2uvw(rgb):
    if rgb[0] == rgb[1] == rgb[2] == 127:
        #return np.array([np.nan, np.nan, np.nan])
        return np.array([0,0,0])
    out = np.zeros_like(rgb).astype(np.float64)
    out[0] = -1.0 + (2*rgb[0] / 255)
    out[1] = -1.0 + (2*rgb[1] / 255)
    out[2] = -1.0 + (2*rgb[2] / 255)
    l = np.linalg.norm(out)
    return out/l


def rgb2uvw(img):
    h,w,_ = img.shape
    out = np.zeros_like(img).astype(np.float64)
    for y in range(h):
        for x in range(w):
            out[y,x] = __rgb2uvw(img[y,x])
    return out

def __uvw2rgb(uvw):
    out = np.zeros_like(uvw).astype(np.uint8)
    out[0] = int(127.5 + 127.5*uvw[0])
    out[1] = int(127.5 + 127.5*uvw[1])
    out[2] = int(127.5 + 127.5*uvw[2])
    return out

def uvw2rgb(img):
    h,w,_ = img.shape
    out = np.zeros_like(img).astype(np.uint8)
    for y in range(h):
        for x in range(w):
            out[y,x] = __uvw2rgb(img[y,x])
    return out

def normalize(img):
    h,w,_ = img.shape
    for y in range(h):
        for x in range(w):
            l = np.linalg.norm(img[y,x])
            if l > 0:
                img[y,x] /= l
    return img

def get_mask(img):
    h,w,d = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            l = np.linalg.norm(img[y,x])
            if l > 0:
                out[y,x] = np.ones(d)
    return out

def get_gradfield(img):
    h,w,d = img.shape
    for y in range(h):
        for x in range(w):
            img[y,x,0] /= img[y,x,2]
            img[y,x,1] /= img[y,x,2]
    return img[:,:,0:2]


def coinid_from_filename(f):
    f1 = os.path.basename(f)
    f2 = re.sub("_202[0-9]", "$", f1).split("$")
    f3 = f2[0].replace("_", ":")
    return f3


def coin_metadata(filename, data_path):
    
    meta = {}

    cid = coinid_from_filename(filename)
    fcid = cid.replace(":", "_")
    filelist = glob.glob(data_path + "**/*" + fcid + "*_meta.json", recursive=True)
    
    for json_file in filelist:
        with open(json_file) as jf:
            json_data = json.load(jf)
            
            if json_data["identifier"] != cid:
                continue
            
            meta["jsonFile"] = json_file
            meta["identifier"] = json_data["identifier"]
            meta["type"] = json_data["type"]
            
            assert(json_data["imageScale"]["unit"] == "mm/pixel")
            meta["ppmm"] = 1.0 / json_data["imageScale"]["value"]
            
    return meta

def __round_odd(x):
    x = int(x)
    if (x % 2) == 1:
        return x
    return x + 1


def cart2pol(x):
    rho = np.linalg.norm(x)
    phi = np.arctan2(x[1], x[0])
    return np.array([rho, phi])
