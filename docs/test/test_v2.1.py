from __future__ import division
import sys, os
import numpy as np

sys.path.append('../../src')
import lib_pytb as lib
#reload(lib)

#path to diam.save file
fpath = '../../'  #PyTB dir

outdir           = os.path.join(fpath,'docs/test_out')     #dir for output data
xml_data_file    = os.path.join(fpath,'docs/test/silicon.save/data-file.xml') #dir input data
atomic_proj_file = os.path.join(fpath,'docs/test/silicon.save/atomic_proj.xml') #dir input data
QE_xml_data_file = os.path.join(outdir,'QE_xml_data.npz') #output file
Hk_outfile       = os.path.join(outdir,'Hk_nonortho.npz')
HR_outfile       = os.path.join(outdir,'HR_ws_nonortho.npz')

nbnds_norm  = 0
shift_type  = 0 #0 regular
shift       = 4 #eV
nproc       = 1

redo_xml    =True  #True, parse the xml file and save it to QE_xml_data.npz
read_eigs   =True
read_S      =True
read_U      =True
redo_Hk     =True
redo_HR     =True
Hk_space    = 'nonortho'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

if redo_xml:
    lib.read_QE_output_xml_v4(
        xml_data_file,
        QE_xml_data_file,
        atomic_proj=atomic_proj_file,
        read_eigs=read_eigs,
        read_U=read_U, 
        read_S=read_S)
    
#get H(k)
if redo_Hk:
    Hk_nonortho = lib.build_Hk_5(
        QE_xml_data_file,
        shift,
        shift_type,
        Hk_space,
        Hk_outfile,
        nbnds_norm=nbnds_norm)

############## get H(R)
if redo_HR:
    nk1=8
    nk2=8
    nk3=8
    
    lib.build_HR_par_6(
        QE_xml_data_file,
        HR_outfile,
        Hk_outfile,
        Hk_space,
        nx=nk1,
        ny=nk2,
        nz=nk3)

############## get interpolated bands 
Erange = [-15,6]

G= [0.000, 0.000, 0.000]    
X= [0.500, 0.500, 0.000]    
L= [0.500, 0.500, 0.500]    
K= [0.750, 0.375, 0.375]    
W= [0.750, 0.500, 0.250]    

Kfrac = [G,X,W,L,G,K,W]
nkmesh= 6*[50] #number of segments * number of points per segment

lib.get_interpolated_bands_3(Kfrac,nkmesh,HR_outfile,Erange)

