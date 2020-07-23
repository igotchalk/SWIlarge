import flopy
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shutil
import time

########## INPUT #############
it = int(sys.argv[1])-1
f_varlist = Path(sys.argv[2])
job_id = sys.argv[3]

# it=0
# f_varlist = Path('../data/PriorModel/varlist.pkl')
# job_id='test'
print(it,f_varlist,job_id)

########## INPUT #############


if sys.platform.lower()=='linux':
    datadir = Path('/scratch/users/ianpg/SWIlarge/data')
    workdir = Path('/scratch/users/ianpg/SWIlarge/work')
    MPSdir = datadir.joinpath('lith/sgems/MPS')
    lithdir = datadir.joinpath('lith/sgems/')
    GISdir = datadir.joinpath('GIS')
    priordir = datadir.joinpath('PriorModel')
    modeldir = datadir.joinpath('NM_model')
elif sys.platform.lower()=='darwin':
    datadir = Path('../data')
    workdir = Path('../work')
    MPSdir = Path('/Users/ianpg/Dropbox/temp_convenience/SWIlarge/data/lith/sgems/MPS')
    GISdir = datadir.joinpath('GIS')
    lithdir = datadir.joinpath('lith/sgems/')
    priordir = datadir.joinpath('PriorModel')
    modeldir = datadir.joinpath('NM_model')

nmgwmdir_empty = datadir.joinpath('nmgwmdir_empty') #<-- removed everything but DIS
nmgwmdir_cal = datadir.joinpath('Calibrated_small') #<-- removed RCH, WEL, GLO, LST from the NAM file to load much faster
figdir = workdir.joinpath('figs')
outputdir = workdir.joinpath('output')

import config
import utils



#%% Useful functions
def load_obj(dirname,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)

def save_obj(dirname,obj,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def make_timestamp(YMD=True,HM=True):
    import datetime
    if YMD:
        ymd = '%Y%m%d'
    else:
        ymd = ''
    if HM:
        hm = '%H%M'
    else:
        hm = ''
    if YMD and HM:
        sep = '_'
    else:
        sep = ''
    return datetime.datetime.now().strftime('{}{}{}'.format(ymd,sep,hm))



if len(f_varlist.name.split('.'))>1:
    nam = f_varlist.name.split('.')[0]
else:
    nam = f_varlist.name
varlist = load_obj(f_varlist.parent,nam)
ts = make_timestamp()



print('copying files...')
model_ws = workdir.joinpath('SV_{}'.format(it))

print('loading model...')
##Loading
modelname = 'SV'
m= flopy.seawat.Seawat.load(modelname + '.nam',exe_name=config.swexe, model_ws=model_ws.as_posix())
rows = np.load(model_ws.joinpath('rows.npy'))
starttime = np.load(model_ws.joinpath('starttime.npy'))
layer_mapping_ind_full = np.load(GISdir.joinpath('layer_mapping_ind_full.npy'))                                 
layer_mapping_ind = layer_mapping_ind_full[:,rows,:]
# m = flopy.seawat.Seawat(modelname, exe_name=config.swexe, model_ws=model_ws.as_posix(),verbose=verbose)
thinmsk_in_aqt = np.load(model_ws.joinpath('thinmsk_in_aqt.npy'))
wellmsk_in_aqt = np.load(model_ws.joinpath('wellmsk_in_aqt.npy'))


print('unpacking and setting new vars...')
##Unpack vars
thinning =varlist['thinning'][it] #done
n_conduits = int(varlist['n_conduits'][it])

por_sand = varlist['por_sand'][it] #done
por_clay = varlist['por_clay'][it] #done
aL = varlist['aL'][it] #done
kvh = varlist['kvh'][it] #done
kh_sand_180 = varlist['kh_sand_180'][it] #done
kh_clay_180 = varlist['kh_clay_180'][it] #done
kh_sand_400 = varlist['kh_sand_400'][it] #done
kh_clay_400 = varlist['kh_clay_400'][it] #done
kh_lay1     = varlist['kh_lay1'][it] #done 
BC_change   = varlist['BC_change'][it] #done

x_cond =  np.random.randint(150,m.ncol-5,size=n_conduits)
y_cond =np.random.randint(rows[0],rows[-1],size=n_conduits)

msh=  np.array([np.meshgrid(np.arange(x-5,x+5),np.arange(y-5,y+5)) for x,y in zip(x_cond,y_cond)])
x_cond=np.ravel(msh[:,0,:,:])
y_cond=np.ravel(msh[:,1,:,:])

extra_conds = np.zeros_like(thinmsk_in_aqt)
extra_conds[:,y_cond,x_cond] = True
extra_conds = np.logical_and(extra_conds,layer_mapping_ind_full==4)


# thinning_msk = thinmsk_in_aqt.copy()
thinning_msk = np.logical_or(thinmsk_in_aqt,extra_conds)
thin =np.round(thinning/25,2)*25
thck = thinmsk_in_aqt.sum(axis=0)
thck_new = np.round((1-thin)*thck,0).astype(np.int)

for lay in range(m.nlay):
    ij = np.argwhere(thinmsk_in_aqt[lay,:,:])
    if len(ij)>0:
        for val in ij:
            thck_val = thinning_msk[:,val[0],val[1]].sum()
            if thck_new[val[0],val[1]]==0:
#                 print('zero thickness now...')
                thinning_msk[:,val[0],val[1]]=False
            else:
#                 print('thinning now...')
                thinning_msk[lay:lay+thck_new[val[0],val[1]],val[0],val[1]]=True
                thinning_msk[:lay,val[0],val[1]]=False
                thinning_msk[lay+thck_new[val[0],val[1]]:,val[0],val[1]]=False        
            


exportdir = outputdir.joinpath('SV')
if not exportdir.exists():
    exportdir.mkdir(parents=True)

    
np.save(exportdir.joinpath('thinning_msk_{job_id}_{it}.npy'.format(job_id=job_id,it=it)),thinning_msk)