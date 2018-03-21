
from __future__ import division
from __future__ import print_function 
import numpy as np
import os
import sys


def read_UPF(fullpath):
    import numpy as np
    import re
    import xml.etree.ElementTree as et
    import scipy.io as sio
    import cPickle as pickle
    from collections import defaultdict
    psp = defaultdict(dict)
   
    Bohr2Angs=0.529177249000000
   
    f1 = open(fullpath,'r')
    f2 = open('tmp.xml','w')
    for line in f1:
        f2.write(line.replace('&','&amp;'))
    f1.close()
    f2.close()
   
    tree = et.parse('tmp.xml')
    root = tree.getroot()
   
    head = root.find('PP_HEADER')
    mesh    =int(head.attrib["mesh_size"])
    nwfc    =int(head.attrib["number_of_wfc"])
    nbeta   =int(head.attrib["number_of_proj"])
   
    psp['PP_HEADER']['number_of_wfc']=nwfc
   
    straux  =head.attrib["is_ultrasoft"]
    if straux.upper()=='.T.' or straux.upper()=='T':
       is_ultrasoft=1
    elif straux.upper()=='.F.' or straux.upper()=='F':
       is_ultrasoft=0
    else:
       sys.exit('is_ultrasoft: String not recognized as boolean %s'%straux)
   
    straux  =head.attrib["is_paw"]
    if straux.upper()=='.T.' or straux.upper()=='T':
       is_paw=1
    elif straux.upper()=='.F.' or straux.upper()=='F':
       is_paw=0
    else:
       sys.exit('is_paw: String not recognized as boolean %s'%straux)
   
    straux  =head.attrib["has_wfc"]
    if straux.upper()=='.T.' or straux.upper()=='T':
       has_wfc=1
    elif straux.upper()=='.F.' or straux.upper()=='F':
       has_wfc=0
    else:
       sys.exit('has_wfc: String not recognized as boolean %s'%straux)
    
    psp["is_ultrasoft"]=is_ultrasoft
    psp["is_paw"]      =is_paw
    psp["has_wfc"]     =has_wfc
   
    for rootaux in root.iter('PP_R'):
        sizeaux = int(rootaux.attrib['size'])
        if mesh != sizeaux:
           sys.exit('Error: The size of PP_R does not match mesh: %i != %i'%(sizeaux,mesh))
        xxaux = re.split('\n| ',rootaux.text)
        rmesh  =np.array(map(float,filter(None,xxaux))) #In Bohrs
        if mesh != len(rmesh):
          sys.exit('Error: wrong mesh size')
    psp['PP_MESH']['mesh']=mesh
    psp['PP_MESH']['PP_R']=rmesh
   
    for rootaux in root.iter('PP_RAB'):
        sizeaux = int(rootaux.attrib['size'])
        if mesh != sizeaux:
           sys.exit('Error: The size of PP_RAB does not match mesh: %i != %i'%(sizeaux,mesh))
        xxaux = re.split('\n| ',rootaux.text)
        rmesh  =np.array(map(float,filter(None,xxaux))) #In Bohrs
        if mesh != len(rmesh):
          sys.exit('Error: wrong mesh size')
    psp['PP_MESH']['PP_RAB']=rmesh
   
    for rootaux in root.iter('PP_LOCAL'):
        sizeaux = int(rootaux.attrib['size'])
        if mesh != sizeaux:
           sys.exit('Error: The size of PP_LOCAL does not match mesh: %i != %i'%(sizeaux,mesh))
        xxaux = re.split('\n| ',rootaux.text)
        local =np.array(map(float,filter(None,xxaux))) #
        if mesh != len(local):
          sys.exit('Error: wrong mesh size')
    psp['PP_LOCAL']=local
    kkbeta = np.zeros(nbeta)
    lll    = np.zeros(nbeta)
    for ibeta in range(nbeta):
        for rootaux in root.iter('PP_BETA.'+str(ibeta+1)):
            kkbeta[ibeta] = int(rootaux.attrib['size'])
            lll[ibeta]    = int(rootaux.attrib['angular_momentum'])
            xxaux         = re.split('\n| ',rootaux.text)
            rmesh         = np.array(map(float,filter(None,xxaux))) 
            if kkbeta[ibeta] != len(rmesh):
              sys.exit('Error: wrong mesh size')
        psp["beta_"+str(ibeta+1)]=rmesh
    psp["nbeta"] =nbeta
    psp["lll"]   =lll
    psp["kkbeta"]=kkbeta
   
    chi = np.zeros((0,mesh))
    pswfc = root.find('PP_PSWFC')
    els = []
    lchi= []
    oc  = []
    for node in pswfc:
      print(node.tag, node.attrib['l'],node.attrib['label'])
      sizeaux = int(node.attrib['size'])
      if sizeaux != mesh: sys.error('Error in mesh size while reading PP_PSWFC info')
   
      els.append(node.attrib['label'])
      lchi.append(int(node.attrib['l']))
      oc.append(float(node.attrib['occupation']))
      xxaux = re.split('\n| ',node.text)
      wfc_aux  =np.array([map(float,filter(None,xxaux))])
      chi = np.concatenate((chi,wfc_aux))
    if nwfc != chi.shape[0]: 
      sys.error('Error: wrong number of PAOs')
    else:
      print('Number of radial wavefunctions found: %i' % chi.shape[0])
    psp['PP_PSWFC']['PP_CHI']       =np.transpose(chi)
    psp['PP_PSWFC']['PP_CHI_label'] =els
    psp['PP_PSWFC']['PP_CHI_l']     =lchi
    psp['PP_PSWFC']['PP_CHI_occupation']  =oc
    psp['PP_PSWFC']['PP_CHI_size']  =mesh
   
   
    if has_wfc:
       rootaux = root.find('PP_FULL_WFC')
       number_of_full_wfc=int(rootaux.attrib['number_of_wfc'])
       psp['PP_FULL_WFC']['number_of_wfc']=number_of_full_wfc
    
       full_aewfc_label=[]
       full_aewfc_l    =[]
       full_aewfc      = np.zeros((0,mesh))
       for ibeta in range(nbeta):
           for rootaux in root.iter('PP_AEWFC.'+str(ibeta+1)):
               print ('Found: '+'PP_AEWFC.'+str(ibeta+1))
               sizeaux       = int(rootaux.attrib['size'])
               if sizeaux != mesh: sys.error('Error in mesh size while reading PP_AEWFC info')
               full_aewfc_l.append(int(rootaux.attrib['l']))
               full_aewfc_label.append(rootaux.attrib['label'])
               xxaux         = re.split('\n| ',rootaux.text)
               rmesh         = np.array([map(float,filter(None,xxaux))]) 
               if sizeaux != rmesh.shape[1]: sys.exit('Error: wrong mesh size')
               full_aewfc    = np.concatenate((full_aewfc,rmesh))
       psp['PP_FULL_WFC']["PP_AEWFC"]       =np.transpose(full_aewfc)
       psp['PP_FULL_WFC']["PP_AEWFC_label"] =full_aewfc_label       
       psp['PP_FULL_WFC']["PP_AEWFC_l"]     =full_aewfc_l       
       psp['PP_FULL_WFC']["PP_AEWFC_size"]  =mesh
   
       full_pswfc_label=[]
       full_pswfc_l    =[]
       full_pswfc      = np.zeros((0,mesh))
       for ibeta in range(nbeta):
           for rootaux in root.iter('PP_PSWFC.'+str(ibeta+1)):
               print('Found: '+'PP_PSWFC.'+str(ibeta+1))
               sizeaux       = int(rootaux.attrib['size'])
               if sizeaux != mesh: sys.error('Error in mesh size while reading PP_PSWFC info')
               full_pswfc_l.append(int(rootaux.attrib['l']))
               full_pswfc_label.append(rootaux.attrib['label'])
               xxaux         = re.split('\n| ',rootaux.text)
               rmesh         = np.array([map(float,filter(None,xxaux))]) 
               if sizeaux != rmesh.shape[1]: sys.exit('Error: wrong mesh size')
               full_pswfc    = np.concatenate((full_pswfc,rmesh))
       psp['PP_FULL_WFC']["PP_PSWFC"]       =np.transpose(full_pswfc)
       psp['PP_FULL_WFC']["PP_PSWFC_label"] =full_pswfc_label       
       psp['PP_FULL_WFC']["PP_PSWFC_l"]     =full_pswfc_l       
       psp['PP_FULL_WFC']["PP_PSWFC_size"]  =mesh
   
    return psp
   
   
       
   
    with open(fullpath + '.p','wb') as fp:
       pickle.dump(psp,fp)
   
    sio.savemat(fullpath + '.mat',psp)
def write_formatted_chi_2(upf_full_filename,outdir,ralpha=None,do_norm=None):

    psp = read_UPF(upf_full_filename)

    outfile = os.path.join(outdir,'PP_CHI.xml')
    print('Printing formatted PP CHI to: {0:s}'.format(outfile))
    fid     = open(outfile,"w")

    occ_str = eformat(0,15,3) #the occupation can be left as 0
    natwfc  = psp['PP_FULL_WFC']['number_of_wfc']
    radials = psp['PP_FULL_WFC']['PP_PSWFC']
    labels  = psp['PP_FULL_WFC']['PP_PSWFC_label']
    l       = psp['PP_FULL_WFC']['PP_PSWFC_l']
    size    = psp['PP_FULL_WFC']['PP_PSWFC_size']
    rab     = psp['PP_MESH']['PP_RAB']
    r       = psp['PP_MESH']['PP_R']

    pseudo_e_str = eformat(0,15,3) #the pseudo energy can be left as 0

    if ralpha is not None:
       if (len(ralpha) != natwfc):
          sys.exit('Number of scaling parameters do not match number of wfcs')
       for ichi in range(natwfc):
           print('scaling wfc %d by exp(-%f *r)'%(ichi,ralpha[ichi]))
           radials[:,ichi] = radials[:,ichi]*np.exp(-ralpha[ichi]*r)

    if do_norm is not None:
       if (len(do_norm) != natwfc):
          sys.exit('Number of normalization flags do not match number of wfcs')
       for ichi in range(natwfc):
           if do_norm[ichi]:
               norm_factor = 1/np.sqrt(sum(radials[:,ichi]**2*rab))
               print('Normalizing wfc %d to 1. Initial normalization was %f'%(ichi,1/norm_factor))
               radials[:,ichi] = radials[:,ichi]*norm_factor      

    for ichi in range(natwfc):
        print('    <PP_CHI.{0:d} type="real" size="{1:d}"'
              ' columns="4" index="{2:d}" label="{3:s}"'
              ' l="{4:d}" occupation="{5:s}" n="{6:d}"\npseudo_energy="{7:s}">'.
              format(ichi+1,size,ichi+1,labels[ichi],l[ichi],occ_str,l[ichi]+1,pseudo_e_str),file=fid)
        chi_str= radial2string(radials[:,ichi])
        print('{0:s}'.format(chi_str),file=fid)
        print('    </PP_CHI.{0:d}>'.format(ichi+1),file=fid)

    fid.close()

    return radials
