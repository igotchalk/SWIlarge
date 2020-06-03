#Post-processing functions
import os
import numpy as np
import flopy
from pathlib import Path


def copy_rename(src_file, dst_file):
    import shutil
    from pathlib import Path
    shutil.copy(str(Path(src_file)), str(Path(dst_file)))
    return

def load_obj(dirname,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'rb') as f:
        return pickle.load(f)

def save_obj(dirname,obj,name):
    import pickle
    with open(Path(dirname).joinpath(name + '.pkl').as_posix(), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def plotdischarge(m,color='w',per=-1,scale=50,rowslice=0,iskip=1):
    import matplotlib.pyplot as plt
    fname = os.path.join(m.model_ws, '' + m.name + '.cbc')
    budobj = flopy.utils.CellBudgetFile(fname)
    qx = budobj.get_data(text='FLOW RIGHT FACE')[per]
    qz = budobj.get_data(text='FLOW LOWER FACE')[per]

    # Average flows to cell centers
    qx_avg = np.empty(qx.shape, dtype=qx.dtype)
    qx_avg[:, :, 1:] = 0.5 * (qx[:, :, 0:m.ncol-1] + qx[:, :, 1:m.ncol])
    qx_avg[:, :, 0] = 0.5 * qx[:, :, 0]
    qz_avg = np.empty(qz.shape, dtype=qz.dtype)
    qz_avg[1:, :, :] = 0.5 * (qz[0:m.nlay-1, :, :] + qz[1:m.nlay, :, :])
    qz_avg[0, :, :] = 0.5 * qz[0, :, :]

    y, x, z = m.dis.get_node_coordinates()
    X, Z = np.meshgrid(x, z[:, 0, 0])

    ax = plt.gca()
    cpatchcollection = ax.quiver(X[::iskip, ::iskip], Z[::iskip, ::iskip],
              qx_avg[::iskip, rowslice, ::iskip], -qz_avg[::iskip, rowslice, ::iskip],
              color=color, scale=scale, headwidth=8, headlength=3,
              headaxislength=1, width=0.0025)
    return cpatchcollection

def permute_kstpkper(ucnobj):
    kstpkper = ucnobj.get_kstpkper()
    kstpkper_unique = []
    index_unique = []
    niter = 0
    for entry in kstpkper:
        if not entry in kstpkper_unique:
            kstpkper_unique.append(entry)
            index_unique.append(niter)
        niter += 1
    return kstpkper_unique, index_unique

def kstpkper_from_time(ucnobj,tottim):
    kstpkpers = ucnobj.get_kstpkper()
    times = ucnobj.get_times()
    timeind = times.index(tottim)
    kstpkper = kstpkpers[timeind]
    return kstpkper

def kstpkper_ind_from_kstpkper(ucnobj,kstpkper=(0,0)):
    kstpkpers = ucnobj.get_kstpkper()
    kstpkper_unique = permute_kstpkper(ucnobj)[0]
    kstpkper_ind = kstpkper_unique.index(kstpkper)
    return kstpkper_ind

#calculate ocean flow from the cell-by-cell budget file
def get_ocean_outflow(m,ocean_coords):
    fname = os.path.join(m.model_ws, m.name + '.cbc')
    budobj = flopy.utils.CellBudgetFile(fname)
    qx = budobj.get_data(text='FLOW RIGHT FACE')[-1]
    qz = budobj.get_data(text='FLOW LOWER FACE')[-1]
    ocean_flow = np.asarray(qz[ocean_coords])
    return ocean_flow

#calculate the ocean outflows from the constant head file
def get_ocean_outflow_chd(m,ocean_bool=None,tot_stp=None):
    if ocean_bool is None:
        ocean_bool = m.ocean_arr
    fname = os.path.join(m.model_ws,m.name + '.cbc')
    budobj = flopy.utils.CellBudgetFile(fname)
    ch = budobj.get_data(text='CONSTANT HEAD')

    #ocean_ind_MF = np.ravel_multi_index(np.where(ocean_bool==1),(m.nlay,m.nrow,m.ncol))+1 #ones-based
    ocean_ind_MF = np.ravel_multi_index(ocean_bool,(m.nlay,m.nrow,m.ncol))+1 #ones-based
    if tot_stp is None:
        tot_stp = len(ch)-1
    #Get flux from .cbc file
    flx = []
    for node,val in ch[tot_stp]:
        if node in ocean_ind_MF:
            flx.append(-val)
    #Assign to grid
    ocean_outflow = np.zeros((m.nlay,m.nrow,m.ncol))
    #ocean_outflow[np.where(ocean_bool==1)] = flx
    ocean_outflow[ocean_bool] = flx
    return ocean_outflow

def get_salt_outflow(m,kstpkper=None):
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    totim = ucnobj.get_times()[-1]
    if kstpkper==None:
        kstpkper = ucnobj.get_kstpkper()[-1]
    ocean_conc = ucnobj.get_data(kstpkper=kstpkper)
    return ocean_conc

def plot_background(mm,array,label=None):
    if label==None:
        label = [ k for k,v in globals().items() if v is array][-1]
    if label=='hk':
        norm=matplotlib.colors.LogNorm()
        vmin=hkClay
        vmax=hkSand
        cmap='jet'
    else:
        norm = None
        vmin=None
        vmax=None
        cmap='jet'
    cpatchcollection = mm.plot_array(array,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    cpatchcollection.set_label(label)
    return cpatchcollection,label

def read_ref(fname='ref_file.txt'):
    import re
    #Load ref file to get info about model
    reffile = os.path.join('.',fname)
    reftext = open(reffile, 'r').read()
    beg = [m.start() for m in re.finditer('<<<', reftext)]
    betw = [m.start() for m in re.finditer('>>>', reftext)]
    end = [m.start() for m in re.finditer('\n', reftext)]
    d = {}
    for i in range(len(beg)):
        d[str(reftext[beg[i]+3:betw[i]])] =  reftext[betw[i]+3:end[i]]
    return d
    #[exec(reftext[beg[i]+3:betw[i]]) for i in range(len(beg))]

def sample_uniform(low, high, shape, logyn):
    '''
    #Samples a uniform distribution the nummber of times shown in 
    low: low value in dist
    high: high value in dist
    shape: shape of samples
    logyn: if True, samples as a log-normal distribution. 
        If False, samples as a uniform distribution. Returned values are *not* in logspace  
    '''

    if logyn:
        log_param_list = np.random.uniform(np.log(low), np.log(high), shape)
        param_list = np.exp(log_param_list)
    else:
        param_list = np.random.uniform(low, high, shape)
    return param_list


def create_varlist(tot_it,heterogenous=1,saveyn=True, ws=Path('./')):
    varlist = {}
    #head_inland_sum
    logyn = False
    low = -1.1
    high = 0
    varlist['head_inland_sum'] = sample_uniform(low, high, tot_it, logyn)
    
    #head_inland_wint
    logyn = False
    low = 0
    high = 3
    varlist['head_inland_wint'] = sample_uniform(low, high, tot_it, logyn)
    
    if heterogenous==0:
    #HOMOGENOUS ONLY
        # log_hk
        logyn = True
        low = 80
        high = 80
        varlist['hk'] = sample_uniform(low, high, tot_it, logyn)
    
        ##por: porosity
        logyn = False
        low = .2
        high = .5
        varlist['por'] = sample_uniform(low, high, tot_it, logyn)
    
    elif heterogenous in [1,2]:
        
        #########HETEROGENOUS ONLY ##############
        
        #CF_glob: global clay-fraction (mu in the random gaussian simulation)
        logyn = False
        low = .1
        high = .9
        varlist['CF_glob'] = sample_uniform(low, high, tot_it, logyn)
        
        #CF_var: variance in clay-fraction (sill in the random gaussian simulation)
        logyn = False
        low = .001
        high = .05
        varlist['CF_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #hk_var: variance in hk (sill in the random gaussisan simulation)
        logyn = True
        low = .005
        high = .175
        varlist['hk_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #seed for random gaussian simulation
        varlist['seed'] = np.arange(1,tot_it+1)
    
        #hk_mean: mean in hk (set constant)
        varlist['hk_mean'] = np.ones(tot_it)*np.log10(50)
        
        #por_mean: global porosity (mu in the random gaussian simulation)
        logyn = False
        low = 0.3
        high = 0.4
        varlist['por_mean'] = sample_uniform(low, high, tot_it, logyn)
        
        #por_var: variance in porosity (sill in the random gaussian simulation)
        logyn = True
        low = .00001
        high = .005
        varlist['por_var'] = sample_uniform(low, high, tot_it, logyn)
        
        #vario_type: model for random gaussian simulation
        varlist['vario_type'] = ['Gaussian' if v==1 else 'Exponential' for v in np.random.randint(0,2,tot_it)]
    
        #corr_len
        logyn = False
        low = 250
        high = 1000
        varlist['corr_len'] = sample_uniform(low, high, tot_it, logyn)
    
        #corr_len_zx
        # equal to lz/lx
        low= .01
        high = .1
        logyn = False
        varlist['corr_len_zx'] = sample_uniform(low, high, tot_it, logyn)
    
    
        #corr_len_yx
        # equal to ly/lx
        low= 0.1
        high = 1
        varlist['corr_len_yx'] = sample_uniform(low, high, tot_it, logyn)
    
        #clay_lyr_yn
        varlist['clay_lyr_yn'] = np.random.randint(0,2,tot_it,dtype=bool)
               
    
    #### END INSERTED BLOCK ########
    
    
    # vka: ratio of vk/hk
    logyn = False
    low = 1 / 20
    high = 1
    varlist['vka'] = sample_uniform(low, high, tot_it, logyn)
    
    # al: #longitudinal dispersivity (m)
    logyn = False
    low = 0.1
    high = 20
    varlist['al'] = sample_uniform(low, high, tot_it, logyn)
    
    # dmcoef: #dispersion coefficient (m2/day)
    #      log-uniform [1e-10,1e-5] #2e-9 from Walther et al
    logyn = True
    low = 1e-10
    high = 1e-5
    varlist['dmcoef'] = sample_uniform(low, high, tot_it, logyn)
    
    # sy: specific yield
    logyn = False
    low = 0.1
    high = 0.4
    varlist['sy'] = sample_uniform(low, high, tot_it, logyn)

    # ss: specific storage
    logyn = False
    low = 5.0e-5
    high = 5.0e-3
    varlist['ss'] = sample_uniform(low, high, tot_it, logyn)
    
    # Wel
    logyn = True
    # low = 1e2
    # high = 1e3
    low = 10
    high = 5e2
    varlist['wel'] = sample_uniform(low, high, (4, tot_it), logyn)
    
    # rech
    logyn = True
    low = 3.5e-4 
    high = 1.5e-3
    varlist['rech'] = sample_uniform(low, high, tot_it, logyn)
    
    # farm rech as a fraction of well extraction
    logyn = False
    low = .05
    high = .20
    varlist['rech_farm'] = sample_uniform(low, high, tot_it, logyn)
    
    # riv_stg
    logyn = False
    low = 0.5
    high = 1.5
    varlist['riv_stg'] = sample_uniform(low, high, tot_it, logyn)
    
    # riv_cond
    logyn = True
    low = 0.1
    high = 100
    varlist['riv_cond'] = sample_uniform(low, high, tot_it, logyn)
    
    #Success log
    varlist['success'] = np.ones(tot_it,dtype=np.int)*-1
    
    varlist['it'] = np.arange(tot_it)
    
    # Save
    save_obj(ws, varlist, 'varlist')
    print('Saved file', Path(ws).joinpath('varlist.pkl'))
    return varlist


def make_timestamp(ymd='%Y%m%d',hm='%H%M'):
    import datetime
    if len(ymd)>0 and len(hm)>0:
        sep = '_'
    else:
        sep = ''
    return datetime.datetime.now().strftime('{}{}{}'.format(ymd,sep,hm))


def create_MC_file(ws):
    ts = make_timestamp()
    MC_dir = Path(ws).joinpath('MC_expt_' + ts)
    if not MC_dir.exists():
        MC_dir.mkdir()
    MC_file = MC_dir.joinpath('expt.txt')
    with MC_file.open('w') as wf:
        wf.close
    print(MC_file)
    return MC_file



