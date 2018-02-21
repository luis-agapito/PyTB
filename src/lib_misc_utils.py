from __future__ import division
from StringIO import StringIO
import numpy as np
import xml.etree.ElementTree as et
import os
import sys
import matplotlib.pyplot as plt

def read_gnuplot_data(filename,sep_str='\n \n'):

  fid=open(filename,'r')
  a=fid.read()
  text=a.split(sep_str)
  nbands = len(text)-1
  print('The separation string %s was used'%repr(sep_str))
  print('Number of blocks found = %d'%nbands)
  if nbands==0:
     return text

  bands = [None]*nbands
  for i in range(nbands):
      bands[i] = np.loadtxt(StringIO(text[i]))

  fid.close()



  return bands
def plot_gnuplot_data(filename,Ef=0,ymin=-10,ymax=10):
    bands = read_gnuplot_data(filename,sep_str='\n \n')

    nbnds = len(bands)
    fig=plt.figure()
    
    for i in range(nbnds):
          x = bands[i][:,0]
          y = bands[i][:,1]
          plt.plot(range(len(x)),y,'.-',linewidth=0.1)

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')

    plt.gca().grid(True)
    plt.ylim(ymin,ymax)
    pdffile = 'eraseplot.pdf'
    print('plot_gnuplot_data: printing to eraseplot.pdf')
    plt.savefig(pdffile,format='pdf')
