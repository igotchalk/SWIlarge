

import flopy
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime

########## INPUT #############
it = int(sys.argv[1])-1
f_varlist = Path(sys.argv[2])
job_id = sys.argv[3]

# it=1
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

print('loading...')
##Loading
modelname = 'NM'
model_ws = workdir.joinpath('NM_{}'.format(it))
m= flopy.seawat.Seawat.load(modelname + '.nam',exe_name=config.swexe, model_ws=model_ws.as_posix())
rows = np.load(model_ws.joinpath('rows.npy'))
starttime = np.load(model_ws.joinpath('starttime.npy'))
layer_mapping_ind_full = np.load(GISdir.joinpath('layer_mapping_ind_full.npy'))                           
kj_lay_red_full = np.load(GISdir.joinpath('KJ_boundaries','kj_lay_red_full.npy'))


layer_mapping_ind = layer_mapping_ind_full[:,rows,:]
kj_lay_red = kj_lay_red_full[:,rows,:]
# m = flopy.seawat.Seawat(modelname, exe_name=config.swexe, model_ws=model_ws.as_posix(),verbose=verbose)


##Make temp folder for writing
model_ws = workdir.joinpath('NM_{}'.format(it))
if not model_ws.exists():
    model_ws.mkdir()
m.model_ws = model_ws.as_posix()


print('unpacking and setting vars...')

##Unpack vars
por_sand = varlist['por_sand'][it] #done
por_clay = varlist['por_clay'][it] #done
aL = varlist['aL'][it] #done
kvh = varlist['kvh'][it] #done
kh_sand_180 = varlist['kh_sand_180'][it] #done
kh_clay_180 = varlist['kh_clay_180'][it] #done
kh_sand_400 = varlist['kh_sand_400'][it] #done
kh_clay_400 = varlist['kh_clay_400'][it] #done
kh_lay1     = varlist['kh_lay1'][it] #done 
DSA_head    = varlist['DSA_head'][it] #done 
DSA_yn      = bool(varlist['DSA_yn'][it]) #done 
BC_change   = varlist['BC_change'][it] #done
hk_aquitard = min(kh_clay_180,kh_clay_400)
hk = np.zeros_like(kj_lay_red,dtype=np.float)
lith_180 = np.load(lithdir.joinpath('snesim','mps180_{}.npy'.format(it))).astype(np.float)[:,rows,:]
lith_400 = np.load(lithdir.joinpath('sisim','sisim400_{}.npy'.format(it))).astype(np.float)[:,rows,:]



lith_180[lith_180==1.] = kh_sand_180
lith_180[lith_180==0.] = kh_clay_180
lith_400[lith_400==1.] = kh_sand_400
lith_400[lith_400==0.] = kh_clay_400


# hk[np.where(layer_mapping_ind_full==0)] = 10000
# hk[np.where(layer_mapping_ind_full==1)] = kh_lay1
# hk[np.where(layer_mapping_ind_full==2)] = hk_aquitard
# hk[np.where(layer_mapping_ind_full==3)] = lith_180[np.where(layer_mapping_ind_full==3)]
# hk[np.where(layer_mapping_ind_full==4)] = hk_aquitard
# hk[np.where(layer_mapping_ind_full==5)] = lith_400[np.where(layer_mapping_ind_full==5)]
# hk[np.where(layer_mapping_ind_full>5)] = 1.

# prsity = np.zeros_like(layer_mapping_ind_full,dtype=np.float)
# prsity[np.isin(hk,(kh_lay1,kh_sand_180,kh_sand_400))]=por_sand
# prsity[np.where(prsity==0.)]=por_clay

hk[np.where(kj_lay_red==0)] = 10000
hk[np.where(kj_lay_red==1)] = kh_lay1
hk[np.where(kj_lay_red==2)] = lith_180[np.where(kj_lay_red==2)] #SVA
# hk[np.where(kj_lay_red==2)] = hk_aquitard
hk[np.where(kj_lay_red==2.5)] = lith_180[np.where(kj_lay_red==2.5)] #int 180
# hk[np.where(kj_lay_red==2.5)] = hk_aquitard

hk[np.where(kj_lay_red==3)] = lith_180[np.where(kj_lay_red==3)]
hk[np.where(kj_lay_red==4)] = lith_180[np.where(kj_lay_red==4)]
# hk[np.where(kj_lay_red==4)] = hk_aquitard
hk[np.where(kj_lay_red==5)] = lith_400[np.where(kj_lay_red==5)]
hk[np.where(kj_lay_red>5)] = 1.

prsity = np.zeros_like(kj_lay_red,dtype=np.float)
prsity[np.isin(hk,(kh_lay1,kh_sand_180,kh_sand_400))]=por_sand
prsity[np.where(prsity==0.)]=por_clay



ghb_data_orig = m.ghb.stress_period_data.data
ghb_data = {}
for per in range(m.dis.nper):
    ghb_per=[]
    for val in ghb_data_orig[per]:
        ghb_per.append([val[0],val[1],val[2],val[3]+BC_change,val[4]])
    ghb_data[per] = ghb_per


if DSA_yn:
    chd_data_orig = m.chd.stress_period_data
    chd_data = {}
    for per in range(m.dis.nper):
        chd_per=[]
        for val in chd_data_orig.data[0]:
            chd_per.append([val[0],val[1],val[2],DSA_head,DSA_head])
        chd_data[per] = chd_per
else: 
    ssm_data_orig = m.ssm.stress_period_data.data
    ssm_data = {}
    for per in range(m.nper):
        ssm_per=ssm_data_orig[per]
        ssm_data[per] = ssm_per[list(ssm_per['itype']!=1)].tolist() 
    ssm = flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data,mxss=len(ssm_data[0])*m.nper)

print('creating objs...')

lpf = flopy.modflow.ModflowLpf(m, hk=hk, vka=kvh, ipakcb=m.lpf.ipakcb,laytyp=0,laywet=0,
                              ss=m.lpf.ss.array,sy=m.lpf.sy.array)

try:
    sconc= m.btn.sconc.array
except:
    sconc= m.btn.sconc[0].array
btn = flopy.mt3d.Mt3dBtn(m,
                         laycon=m.btn.laycon.array, htop=m.btn.htop.array,
                         dz=m.dis.thickness.get_value(), prsity=prsity, icbund=m.btn.icbund.array,
                         sconc=sconc, nprs=1,timprs=m.btn.timprs)

dsp = flopy.mt3d.Mt3dDsp(m, al=aL,dmcoef=2.0e-9)
ghb = flopy.modflow.ModflowGhb(m, stress_period_data=ghb_data)


if DSA_yn:
    chd = flopy.modflow.ModflowChd(m, stress_period_data=chd_data)
else:
    try:
        m.remove_package("CHD")
    except:
        pass

writeyn= True
runyn = True
#Write input
if writeyn:
    print('writing...')
    m.write_input()
    
    
# Try to delete the output files, to prevent accidental use of older files
f_delete = [os.path.join(m.model_ws,'MT3D.CNF'),
            os.path.join(m.model_ws,'MT3D001.MAS'),
            os.path.join(m.model_ws, 'MT3D001.UCN'),
            os.path.join(m.model_ws, modelname + '.hds'),
            os.path.join(m.model_ws, modelname + '.cbc')]

for f in f_delete:
    try:
        os.remove(f)
    except:
        pass

#%%

if runyn:
    print('running...')
    v = m.run_model(silent=False, report=True)
    for idx in range(-3, 0):
        print(v[1][idx])
else:
    print('Not running model!')

exportdir = outputdir.joinpath('NM')
if not exportdir.exists():
    exportdir.mkdir(parents=True)

date_per = starttime + np.cumsum(m.dis.perlen.array)/365
survey_date = 2017.25
survey_kper = np.argmin(np.abs(date_per-survey_date))
end_date = 2040
end_kper = np.argmin(np.abs(date_per-end_date))




fname = os.path.join(m.model_ws, 'MT3D001.UCN')
totim = flopy.utils.binaryfile.UcnFile(fname).get_times()[-1]
try:
    conc_fname = 'conc{}_{}_totim{}_{}.UCN'.format(
        it, ts, str(int(totim)),job_id)
except:
    conc_fname = 'conc{}_{}_totim{}.UCN'.format(
        it, ts, str(int(totim)))

utils.copy_rename(fname,
                 exportdir.joinpath(conc_fname))
conc = flopy.utils.binaryfile.UcnFile(fname).get_data(kstpkper=(0,survey_kper))
np.save(exportdir.joinpath(conc_fname[:-4] + '.npy'),conc)
conc = flopy.utils.binaryfile.UcnFile(fname).get_data(kstpkper=(0,end_kper))
np.save(exportdir.joinpath(conc_fname[:-4] + 'end.npy'),conc)