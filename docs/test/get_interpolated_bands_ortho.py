from __future__ import division
import sys
import scipy.linalg as sla
import os
import numpy as np

sys.path.append('../../src')
import PyTB_lib as lib
reload(lib)

#path to diam.save file
fpath = '../../'  #PyTB dir

outdir           = os.path.join(fpath,'docs/test_out')     #dir for output data
xml_data_path    = os.path.join(fpath,'docs/test/silicon.save') #dir input data
QE_xml_data_file = os.path.join(outdir,'QE_xml_data.npz') #output file
Hk_outfile       = os.path.join(outdir,'Hk_ortho.npz')
HR_outfile       = os.path.join(outdir,'HR_ws_ortho.npz')

nbnds_norm  = 8
shift_type  = 1
shift       = 4 #eV
nproc       = 1

redo_xml    =True  #True, parse the xml file and save it to QE_xml_data.npz
read_eigs   =True
read_S      =True
read_U      =True
redo_Hk     =True
redo_HR     =True

Hk_space    = 'ortho'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

if redo_xml:
    #U,Sks, my_eigsmat, alat, a_vectors, 
    #nkpnts, nspin, kpnts, kpnts_wght, nbnds, 
    #Efermi, nawf)
    #mytuple = lib.read_QE_output_xml_v2(xml_data_path,read_eigs=read_eigs, read_U=read_U, read_S=read_S, nproc=nproc)
    #print mytuple
    #sys.exit('exit')
    U,Sks, eigsmat, alat, a_vectors,\
    nkpnts, nspin, kpnts, kpnts_wght, nbnds, \
    Efermi, nawf = lib.read_QE_output_xml_serial_v3(xml_data_path,QE_xml_data_file,read_eigs=read_eigs, read_U=read_U, read_S=read_S, nproc=nproc)
else:
    data      = np.load(QE_xml_data_file)
    nawf      = int(data['nawf'])
    nkpnts    = int(data['nkpnts'])
    nspin     = int(data['nspin'])
    a_vectors = data['a_vectors']
    alat      = data['alat']
    eigsmat   = data['eigsmat']
    Sks       = data['Sk']
    U         = data['U']
    
#get H(k)
if redo_Hk:
    Hk_ortho = lib.build_Hk_4(nawf,nkpnts,nspin,shift,eigsmat,shift_type,U,Hk_space,Hk_outfile,Sks=Sks,nbnds_norm=nbnds_norm)
else:
    Hk_ortho = np.load(Hk_outfile)['Hks']


############## get H(R)
if redo_HR:
    nk1=8
    nk2=8
    nk3=8
    
    #Wigner-Seitz cell
    HR_ws_mat, _ ,_   = lib.build_HR_serial(nk1,nk2,nk3,QE_xml_data_file,HR_outfile,Hk_ortho,Hk_space,nproc=nproc,cell_type='wigner-seitz')

############## get interpolated bands 
Erange = [-15,6]

G= [0.000, 0.000, 0.000]    
X= [0.500, 0.500, 0.000]    
L= [0.500, 0.500, 0.500]    
K= [0.750, 0.375, 0.375]    
W= [0.750, 0.500, 0.250]    

Kfrac = [G,X,W,L,G,K,W]
nkmesh= 6*[50] #number of segments * number of points per segment

#Wigner-Seitz cell
lib.get_interpolated_bands_3(Kfrac,nkmesh,HR_outfile,Erange)

