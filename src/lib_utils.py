from __future__ import division
import os
import sys
import numpy as np
import multiprocessing
import signal
import time
import datetime

from scipy import linalg as sla
from numpy import linalg as  la
from functools import partial

def create_kpaths(nkmesh,K):
    
    nKpoints = len(K)
    if len(nkmesh) != nKpoints-1: sys.exit('size of nkmesh does not agree with number of kpaths')
    list_aux = []
    for ipath in range(nKpoints-1):
        K1  = K[ipath]
        K2  = K[ipath+1]
        lvec= linspace_vector(K1,K2,nkmesh[ipath]+1)
        if ipath < nKpoints-1-1:
            lvec = lvec[:-1,:]
        list_aux = list_aux + lvec.tolist()
    return np.array(list_aux),np.cumsum([0]+nkmesh)
def linspace_vector(v1,v2,ndivs):
    lx = np.reshape(np.linspace(v1[0],v2[0],ndivs),(ndivs,1))
    ly = np.reshape(np.linspace(v1[1],v2[1],ndivs),(ndivs,1))
    lz = np.reshape(np.linspace(v1[2],v2[2],ndivs),(ndivs,1))
    return np.hstack((lx,ly,lz))
def fname():
    return  sys._getframe(1).f_code.co_name
