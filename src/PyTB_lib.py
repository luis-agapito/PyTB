from __future__ import print_function
#from scipy import linalg
from scipy import linalg as LA
import numpy as np
import xml.etree.ElementTree as ET
import sys
import re

#TODO
#switch U to be saved its transpose .
#print the calculation parameters in the pdf 

#units
Ry2eV   = 13.60569193

def build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U):
    Hks = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            my_eigs=my_eigsmat[:,ik,ispin]
            #Building the Hamiltonian matrix
            #print('Iteration ispin,ik={0:d}{1:d}'.format(ispin,ik))
            E = np.diag(my_eigs)
            UU = np.transpose(U[:,:,ik,ispin]) #transpose of U. Now the columns of UU are the eigenvector of length nawf
            norms = 1/np.sqrt(np.real(np.sum(np.conj(UU)*UU,axis=0)))
            UU[:,:nbnds_norm] = UU[:,:nbnds_norm]*norms[:nbnds_norm]
            eta=shift
            ac = UU[:,:bnd]
            ee1 = E[:bnd,:bnd]
            if bnd == nbnds:
                bd = np.zeros((nawf,1))
                ee2 = 0
            else:
                bd = UU[:,bnd:nbnds]
                ee2= E[bnd:nbnds,bnd:nbnds]
            if shift_type ==0:
                #option 1
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(np.conj(ac).T))
            elif shift_type==1:
                #option 2.
                aux_p=LA.inv(np.dot(np.conj(ac).T,ac))
                Hks[:,:,ik,ispin] = ac.dot(ee1).dot(np.conj(ac).T) + eta*(np.identity(nawf)-ac.dot(aux_p).dot(np.conj(ac).T))
            else:
                sys.exit('shift_type not recognized')
    return Hks

def read_QE_output_xml(fpath,read_S):
 atomic_proj = fpath+'/atomic_proj.xml'
 data_file   = fpath+'/data-file.xml'

 # Reading data-file.xml
 tree  = ET.parse(data_file)
 root  = tree.getroot()

 alatunits  = root.findall("./CELL/LATTICE_PARAMETER")[0].attrib['UNITS']
 alat   = float(root.findall("./CELL/LATTICE_PARAMETER")[0].text.split()[0])

 print("The lattice parameter is: alat= {0:f} ({1:s})".format(alat,alatunits))

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a1")[0].text.split()
 a1=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a2")[0].text.split()
 a2=[float(i) for i in aux]

 aux=root.findall("./CELL/DIRECT_LATTICE_VECTORS/a3")[0].text.split()
 a3=[float(i) for i in aux]

 a_vectors = np.array([a1,a2,a3]) #in Bohrs
 print(a_vectors)


 # Reading atomic_proj.xml
 tree  = ET.parse(atomic_proj)
 root  = tree.getroot()

 nkpnts = int(root.findall("./HEADER/NUMBER_OF_K-POINTS")[0].text.strip())
 print('Number of kpoints: {0:d}'.format(nkpnts))

 nspin  = int(root.findall("./HEADER/NUMBER_OF_SPIN_COMPONENTS")[0].text.split()[0])
 print('Number of spin components: {0:d}'.format(nspin))

 kunits = root.findall("./HEADER/UNITS_FOR_K-POINTS")[0].attrib['UNITS']
 print('Units for the kpoints: {0:s}'.format(kunits))

 aux = root.findall("./K-POINTS")[0].text.split()
 kpnts  = np.array([float(i) for i in aux]).reshape((nkpnts,3))
 print('Read the kpoints')

 aux = root.findall("./WEIGHT_OF_K-POINTS")[0].text.split()
 kpnts_wght  = np.array([float(i) for i in aux])

 if kpnts_wght.shape[0] != nkpnts:
 	sys.exit('Error in size of the kpnts_wght vector')
 else:
 	print('Read the weight of the kpoints')


 nbnds  = int(root.findall("./HEADER/NUMBER_OF_BANDS")[0].text.split()[0])
 print('Number of bands: {0:d}'.format(nbnds))

 aux    = root.findall("./HEADER/UNITS_FOR_ENERGY")[0].attrib['UNITS']
 print('The unit for energy are {0:s}'.format(aux))

 Efermi = float(root.findall("./HEADER/FERMI_ENERGY")[0].text.split()[0])*Ry2eV
 print('Fermi energy: {0:f} eV (only if the above is Rydberg'.format(Efermi))

 nawf   =int(root.findall("./HEADER/NUMBER_OF_ATOMIC_WFC")[0].text.split()[0])
 print('Number of atomic wavefunctions: {0:d}'.format(nawf))

 #Read eigenvalues and projections

 U = np.zeros((nbnds,nawf,nkpnts,nspin),dtype=complex)
 my_eigsmat = np.zeros((nbnds,nkpnts,nspin))
 for ispin in range(nspin):
   for ik in range(nkpnts):
     #Reading eigenvalues
     if nspin==1:
         eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].attrib['type']
     else:
         eigk_type=root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].attrib['type']
     if eigk_type != 'real':
       sys.exit('Reading eigenvalues that are not real numbers')
     if nspin==1:
       eigk_file=np.array([float(i) for i in root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG".format(ik+1))[0].text.split()])
     else:
       eigk_file=np.array([float(i) for i in root.findall("./EIGENVALUES/K-POINT.{0:d}/EIG.{1:d}".format(ik+1,ispin+1))[0].text.split().split()])
     my_eigsmat[:,ik,ispin] = np.real(eigk_file)*Ry2eV-Efermi #meigs in eVs and wrt Ef

     #Reading projections
     for iin in range(nawf): #There will be nawf projections. Each projector of size nbnds x 1
       if nspin==1:
         wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].attrib['type']
         aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/ATMWFC.{1:d}".format(ik+1,iin+1))[0].text
       else:
         wfc_type=root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,iin+1))[0].attrib['type']
         aux     =root.findall("./PROJECTIONS/K-POINT.{0:d}/SPIN.{1:d}/ATMWFC.{2:d}".format(ik+1,ispin+1,iin+1))[0].text

       aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

       if wfc_type=='real':
         wfc = aux.reshape((nbnds,1))#wfc = nbnds x 1
         U[:,iin,ik,ispin] = wfc[:,0]
       elif wfc_type=='complex':
         wfc = aux.reshape((nbnds,2))
         U[:,iin,ik,ispin] = wfc[:,0]+1j*wfc[:,1]
       else:
         sys.exit('neither real nor complex??')

 if read_S:
   Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
   for ik in range(nkpnts):
     #There will be nawf projections. Each projector of size nbnds x 1
     ovlp_type = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].attrib['type']
     aux = root.findall("./OVERLAPS/K-POINT.{0:d}/OVERLAP.1".format(ik+1))[0].text
     aux = np.array([float(i) for i in re.split(',|\n',aux.strip())])

     if ovlp_type !='complex':
       sys.exit('the overlaps are assumed to be complex numbers')
     if len(aux) != nawf**2*2:
       sys.exit('wrong number of elements when reading the S matrix')

     aux = aux.reshape((nawf**2,2))
     ovlp_vector = aux[:,0]+1j*aux[:,1]
     Sks[:,:,ik] = ovlp_vector.reshape((nawf,nawf))
   return(U,Sks, my_eigsmat, alat, a_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf)
 else:
   return(U, my_eigsmat, alat, a_vectors, nkpnts, nspin, kpnts, kpnts_wght, nbnds, Efermi, nawf)

def plot_compare_TB_DFT_eigs(Hks,my_eigsmat):
    import matplotlib.pyplot as plt
    import os

    nawf,nawf,nkpnts,nspin = Hks.shape
    nbnds_tb = nawf
    E_k = np.zeros((nbnds_tb,nkpnts,nspin))

    ispin = 0 #plots only 1 spin channel
    #for ispin in range(nspin):
    for ik in range(nkpnts):
        eigval,_ = LA.eig(Hks[:,:,ik,ispin])
        E_k[:,ik,ispin] = np.sort(np.real(eigval))

    fig=plt.figure
    nbnds_dft,_,_=my_eigsmat.shape
    for i in range(nbnds_dft):
        #print("{0:d}".format(i))
        yy = my_eigsmat[i,:,ispin]
        if i==0:
          plt.plot(yy,'-',linewidth=3,color='lime',label='DFT')
        else:
          plt.plot(yy,'-',linewidth=3,color='lime')

    for i in range(nbnds_tb):
        yy = E_k[i,:,ispin]
        if i==0:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None',label='TB')
        else:
          plt.plot(yy,'ok',markersize=2,markeredgecolor='None')

    plt.xlabel('k-points')
    plt.ylabel('Energy - E$_F$ (eV)')
    plt.legend()
    plt.title('Comparison of TB vs. DFT eigenvalues')
    plt.savefig('comparison.pdf',format='pdf')
    #os.system('open comparison.pdf') #for macs
