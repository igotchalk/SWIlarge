
#Name model

import os
import sys
from pathlib import Path
import numpy as np
import flopy
import SGD
#import config
import datetime
import simulationFFT
import utils

modelname = 'heterog_1000'

###### INPUT ########
it = int(sys.argv[1])
MC_file = Path(sys.argv[2])
basecase_ws = Path(sys.argv[3])
# swexe = '/home/groups/rknight/swtv4'
# it = 0
# MC_file = Path('../work/heterog_1000/MC_expt_20200202_1427/expt.txt')
# basecase_ws =  Path('../work/homog/')
#####################
fpath_basecase = basecase_ws.joinpath(basecase_ws.parts[-1] + '.nam').as_posix()
swexe = '/home/groups/rknight/swtv4'

print(os.getcwd())
print('Running iteration {} \n'
    'MC_file: {} \n'
    'basecase workspace: {} \n'.format(it,MC_file,basecase_ws))



# repo = Path('/scratch/users/ianpg/henry')
#repo = Path('/Users/ianpg/Documents/ProjectsLocal/SWIsmall')
repo = Path('/scratch/users/ianpg/SWISmall/')
workdir = repo.joinpath('work')
figdir = workdir.joinpath('figs')
datadir = repo.joinpath('data')
objdir = repo.joinpath('data', 'objs')
model_ws = repo.joinpath('work', modelname)    
for p in (workdir,figdir,datadir,objdir,model_ws):
    if not p.exists():
        p.mkdir()

#Load model
m = flopy.seawat.Seawat.load(fpath_basecase)
SGD.ModelSGD.Seawat2SGD(m)  #convert to subclass ModelSGD
m.MC_file = MC_file
model_ws = MC_file.parent.joinpath('tmp{}'.format(it))
if not model_ws.exists():
    model_ws.mkdir()
m.model_ws = model_ws.as_posix()
m.name  = modelname

m.exe_name = swexe
nrow,ncol,nlay,nper = m.nrow_ncol_nlay_nper
henry_top,henry_botm = m.dis.top, m.dis.botm
delr,delc = m.dis.delr.array[0], m.dis.delc.array[0]

Lx = 3000.
Ly = 600.
Lz = 80.

henry_top = 3
ocean_elev = 0
delv_first = 5


botm_first = henry_top - delv_first

nlay = int(Lz * 1 / 3)
nrow = int(Ly * (1 / 30))
ncol = int(Lx * (1 / 30))

delv = (Lz - delv_first) / (nlay - 1)
delr = Lx / ncol
delc = Ly / nrow

henry_botm = np.hstack(([botm_first], np.linspace(
    botm_first - delv, henry_top - Lz, nlay - 1)))
delv_vec = np.hstack((delv_first, np.repeat(delv, nlay - 1)))
delv_weight = [x / np.sum(delv_vec) for x in delv_vec]


# Period data
nyrs = 20
Lt = 360 * nyrs  # Length of time in days
perlen = list(np.repeat(180, int(Lt / 180)))
nstp = list(np.ones(np.shape(perlen), dtype=int))

nper = len(perlen)
steady = [False for x in range(len(perlen))]  # Never steady
itmuni = 4  # time unit 4= days
lenuni = 2  # length unit 2 = meter
tsmult = 1.8
ssm_data = None
verbose = False
kper_odd = list(np.arange(1, nper, 2))
kper_even = list(np.arange(0, nper, 2))


# Variable density parameters
Csalt = 35.0001
Cfresh = 0.
densesalt = 1025.
densefresh = 1000.
denseslp = (densesalt - densefresh) / (Csalt - Cfresh)

#Load varlist
varlist = utils.load_obj(m.MC_file.parent,'varlist')
right_edge = utils.load_obj(basecase_ws, 'right_edge')
farm_loc_list = utils.load_obj(basecase_ws,'farm_loc_list')
farm_orig = [(1, 50), (9, 50), (1, 58), (9, 58)]
n_wells = len(farm_orig)
wel_cells = utils.load_obj(basecase_ws,'wel_cells')
riv_loc = utils.load_obj(basecase_ws,'riv_loc')

# sys.path.append(repo.joinpath('notebook').as_posix())
#sw_exe = config.swexe  # set the exe path for seawat
print('Model workspace:', model_ws)

def normal_transform(data1,mu1,mu2,sig1,sig2):
    a = sig2/sig1
    b = mu2 - mu1 * a
    return a*data1 + b

def record_salinity(m,totim=None,writeyn=False,fname_write=None,ts_hms=None):
    if ts_hms is None:
        ts_hms = datetime.datetime.now().strftime('%H-%M-%S')
    # Extract final timestep salinity
    fname = os.path.join(m.model_ws, 'MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(fname)
    if totim is None:
        totim = ucnobj.get_times()[-1]
    conc = ucnobj.get_data(totim=totim)
    if writeyn:
        if fname_write is None:
            fname_write = m.MC_file.parent.joinpath('conc_' + str(int(totim)) + '_' + ts_hms + '.npy')
        print(fname_write)
        np.save(fname_write,conc)
    return conc

def get_hds(m,kstpkper=None):
    f_hds = Path(m.name + '.hds')
    hdsobj = flopy.utils.binaryfile.HeadFile(f_hds.as_posix())
    if kstpkper is None:
        kstpkper = hdsobj.get_kstpkper()[-1]
    return hdsobj.get_data(kstpkper=kstpkper)

def get_base_hds_conc(ws):
    f_ucn = ws.joinpath('MT3D001.UCN')
    ucnobj = flopy.utils.binaryfile.UcnFile(f_ucn.as_posix())
    kstpkperu = ucnobj.get_kstpkper()[-1]

    f_hds = ws.joinpath(ws.parts[-1] + '.hds')
    hdsobj = flopy.utils.binaryfile.HeadFile(f_hds.as_posix())
    kstpkperh = hdsobj.get_kstpkper()[-1]
    return hdsobj.get_data(kstpkper=kstpkperh), ucnobj.get_data(kstpkper=kstpkperu), 

def get_ocean_right_edge(m, ocean_line_tuple, startlay=None, col=None):
    import numpy as np
    point_list = []

    if col is None:
        col = m.ncol - 1
    # If there is no vertical side boundary, return bottom-right corner node
    if len(ocean_line_tuple) == 0:
        if startlay is None:
            startlay = 0
    elif max(ocean_line_tuple[0]) == m.nlay:
        startlay = m.nlay
    elif max(ocean_line_tuple[0]) < m.nlay:
        startlay = max(ocean_line_tuple[0])
    for lay in range(startlay, m.nlay):
        for row in range(m.nrow):
            point_list.append((lay, row, col))
    point_list = tuple(np.array(point_list).T)
    return point_list

def find_nearest(array, value):
    import numpy as np
    idx = (np.abs(array - value)).argmin()
    idx.astype('int')
    return array[idx]

def make_bc_dicts(head_inland_sum_wint):
    #Ocean and inland boundary types
    bc_right_edge='GHB'
    bc_inland = 'GHB'
    ocean_elev =0
    hkSand = 100

    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    chd_data = {}
    ssm_data = {}
    ghb_data = {}
    wel_data = {}
    for i in range(nper):
        dat_chd = []
        dat_ssm = []
        dat_ghb = []
        dat_wel = []
        #Ocean boundary
        # if ocean_hf:
        #     for j in range(np.size(ocean_hf[0])):
        #         if bc_ocean=='CHD':
        #             #CHD: {stress_period: [lay,row,col,starthead,endhead]}
        #             dat_chd.append([ocean_line_tuple[0][j],
        #                         ocean_line_tuple[1][j],
        #                         ocean_line_tuple[2][j],
        #                         ocean_shead[i],
        #                         ocean_ehead[i]])
        #             #SSM: {stress_period: [lay,row,col,concentration,itype]}
        #             dat_ssm.append([ocean_line_tuple[0][j],
        #                         ocean_line_tuple[1][j],
        #                         ocean_line_tuple[2][j],
        #                         Csalt,
        #                         itype['CHD']])
        #         elif bc_ocean=='GHB':
        #             #GHB: {stress period: [lay,row,col,head level,conductance]}
        #             #conductance c = K*A/dL; assume horizontal flow at outlet,
        #             #and calculate length to be at edge of ocean cell, as opposed to mipoint
        #             # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
        #             dat_ghb.append([ocean_hf[0][j],
        #                            ocean_hf[1][j],
        #                            ocean_hf[2][j],
        #                            #ocean_hf[3][j],
        #                             ocean_elev,
        #                            2*hkSand*delc*delv_vec[ocean_hf[0][j]]/delr])
        #             #SSM: {stress_period: [lay,row,col,concentration,itype]}
        #             dat_ssm.append([ocean_hf[0][j],
        #                            ocean_hf[1][j],
        #                            ocean_hf[2][j],
        #                            Csalt,
        #                            itype['GHB']])
        if False:
        	pass
        else:
            pass
        #Right edge boundary
        if bc_right_edge=='GHB':
            for j in range(np.size(right_edge[0])):
                #GHB: {stress period: [lay,row,col,head level,conductance]}
                #conductance c = K*A/dL; assume horizontal flow at outlet,
                #and calculate length to be at edge of ocean cell, as opposed to mipoint
                # c = (K*dy*dz)/(dx/2) = 2*K*delr*delv/delc
                dat_ghb.append([right_edge[0][j],
                               right_edge[1][j],
                               right_edge[2][j],
                               #ocean_hf[3][j],
                                ocean_elev,
                               2*hkSand*delc*delv_vec[right_edge[0][j]]/delr])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                dat_ssm.append([right_edge[0][j],
                               right_edge[1][j],
                               right_edge[2][j],
                               Csalt,
                               itype['GHB']])
        else:
            pass
        #Inland boundary
        if bc_inland=='GHB':
            if i in kper_odd:
                head_inland = head_inland_sum_wint[0]
            elif i in kper_even:
                head_inland = head_inland_sum_wint[1]
            left_edge = get_ocean_right_edge(m,tuple([]),
                  int(np.where(henry_botm==find_nearest(henry_botm,head_inland))[0]),
                col=0)
            for j in range(np.size(left_edge[0])):
                dat_ghb.append([left_edge[0][j],
                               left_edge[1][j],
                               left_edge[2][j],
                                head_inland,
                               2*hkSand*delc*delv_vec[left_edge[0][j]]/delr])
                #SSM: {stress_period: [lay,row,col,concentration,itype]}
                dat_ssm.append([left_edge[0][j],
                               left_edge[1][j],
                               left_edge[2][j],
                               Cfresh,
                               itype['GHB']])
        chd_data[i] = dat_chd
        ssm_data[i] = dat_ssm
        ghb_data[i] = dat_ghb
        wel_data[i] = dat_wel

    #saving concentrations at specified times
    #timprs = [k for k in range(1,np.sum(perlen),50)]
    return chd_data, ssm_data, ghb_data, wel_data

def add_pumping_wells(wel_data,ssm_data,n_wells,flx,rowcol,kper):
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    new_weldata = wel_data
    new_ssmdata = ssm_data
    wel_cells = []
    for k in range(n_wells):
        row,col = rowcol[k]
        for i in range(nper):
            if i in kper:
                for j in range(nlay):
                    #WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append([j,row,col,-flx[k]*delv_weight[j]])
                    wel_cells.append((j,row,col))
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    new_ssmdata[i].append([j,row,col,Cfresh,itype['WEL']]) #since it's a sink, conc. doesn't matter
            else:
                for j in range(nlay):
                    #WEL {stress_period: [lay,row,col,flux]}
                    new_weldata[i].append([j,row,col,0])
                    #SSM: {stress_period: [lay,row,col,concentration,itype]}
                    new_ssmdata[i].append([j,row,col,Cfresh,itype['WEL']]) #since it's a sink, conc. doesn't matter
                    wel_cells.append((j,row,col))
                continue
    wel_cells = tuple(np.array(list(set(wel_cells))).T)
    return new_weldata, new_ssmdata,wel_cells

def write_river_data(riv_loc, stage, cond, riv_grad, kper, ssm_data):

    ####ADD RIVER DATA####
    rbot_vec = np.linspace(riv_grad * Lx, ocean_elev, ncol)

    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    riv_data = {}
    new_ssm_data = ssm_data
    for i in range(nper):
        dat_riv = []
        if i in kper:
            for j in range(np.size(riv_loc[0])):
                # RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                riv_loc[1][j],
                                riv_loc[2][j],
                                stage + rbot_vec[riv_loc[2][j]],
                                cond,
                                rbot_vec[riv_loc[2][j]]])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        else:
            for j in range(np.size(riv_loc[0])):
                # RIV: {stress_period:[lay, row, col, stage, cond, rbot],...}
                dat_riv.append([riv_loc[0][j],
                                riv_loc[1][j],
                                riv_loc[2][j],
                                # set stage as bottom of river
                                rbot_vec[riv_loc[2][j]],
                                cond,
                                rbot_vec[riv_loc[2][j]]])
                # SSM: {stress_period: [lay,row,col,concentration,itype]}
                new_ssm_data[i].append([riv_loc[0][j],
                                        riv_loc[1][j],
                                        riv_loc[2][j],
                                        Cfresh,
                                        itype['RIV']])
        riv_data[i] = dat_riv
    return riv_data, new_ssm_data


def update_run_model(varlist, it, m=m, homogenous=2, runyn=True,
                     plotyn=False,silent=True,start_basecase=True,
                     f_basecase=None,pooling=True,output=None,results=[]):
    # Make timestamp
    ts = utils.make_timestamp()
    print('Running it {} at time {}'.format(it,ts))
    
    if start_basecase:
        strt,sconc = get_base_hds_conc(basecase_ws)
    
    if pooling:
        model_ws_orig = Path(m.model_ws).as_posix() + ''
        tmp = Path(model_ws).joinpath('tmp{}'.format(it))
        if not tmp.exists():
            tmp.mkdir()
        m.model_ws = tmp.as_posix()
        print('temp ws', m.model_ws)

    # unpack values from varlist
    vka = varlist['vka'][it]
    al = varlist['al'][it]
    dmcoef = varlist['dmcoef'][it]
    ss = varlist['ss'][it]
    sy = varlist['sy'][it]

    riv_stg = varlist['riv_stg'][it]
    riv_cond = varlist['riv_cond'][it]
    head_inland_sum = varlist['head_inland_sum'][it]
    head_inland_wint = varlist['head_inland_wint'][it]
    wel = varlist['wel'][:,it]
    rech_farm_pct = varlist['rech_farm'][0]
    farm_size = (200,200)
    rech_farm = [rech_farm_pct*flx/np.prod(farm_size) for flx in wel]
    rech_precip = varlist['rech'][it]
    
    CF_glob = varlist['CF_glob'][it]
    CF_var = varlist['CF_var'][it]
    seed = varlist['seed'][it]
    hk_mean = varlist['hk_mean'][it]
    hk_var = varlist['hk_var'][it]
    por_mean = varlist['por_mean'][it]
    por_var = varlist['por_var'][it]
    corr_len = varlist['corr_len'][it]
    corr_len_yx = varlist['corr_len_yx'][it]
    corr_len_zx = varlist['corr_len_zx'][it]
    clay_lyr_yn = varlist['clay_lyr_yn'][it]
    vario_type = varlist['vario_type'][it]
    


    #set ghb data and create dicts
    chd_data, ssm_data_base, ghb_data, wel_data_base = make_bc_dicts((head_inland_sum,head_inland_wint))
    utils.save_obj(m.MC_file.parent,wel_data_base,'wel_data_base')
    utils.save_obj(m.MC_file.parent,ssm_data_base,'ssm_data_base')

    ssm_data = {}
    # write recharge data
    
    rech_farm_mat = np.zeros((nrow,ncol),dtype=np.float32)
    for i in range(len(rech_farm)):
        rech_farm_mat[farm_loc_list[i]] = rech_farm[i]
    
    rech_data = {}
    for i in range(len(perlen)):
        if i in kper_even:
            rech_data[i] = rech_precip
        elif i in kper_odd:
            rech_data[i] = rech_farm_mat


    # write wel data
    # ssm_data_base = load_obj(m.MC_file.parent, 'ssm_data_base')
    # wel_data_base = load_obj(m.MC_file.parent, 'wel_data_base')
    wel_data, ssm_data, wel_cells = add_pumping_wells(wel_data_base,
                                                      ssm_data_base,
                                                      n_wells,flx=wel,
                                                      rowcol=farm_orig,
                                                      kper=kper_odd)
    if homogenous==1:
        CF_grid = 1
        hk_grid = 10**hk_mean
        por_grid = .4
    elif homogenous==2:
        #Create Gaussian Simulation
        lcol = int(corr_len/delr)
        llay = int(corr_len*corr_len_zx/np.mean(delv))
        lrow = int(corr_len*corr_len_yx/delc)
    #     fft_grid = np.exp(simulationFFT.simulFFT(nrow, nlay, ncol, mu, sill, vario_type, lrow , llay, lcol))   
        CF_grid = simulationFFT.simulFFT(nrow,nlay, ncol,CF_glob,CF_var,vario_type, lrow , llay, lcol,seed=seed)
        hk_grid = 10**normal_transform(CF_grid,CF_glob,hk_mean,np.sqrt(CF_var),np.sqrt(hk_var))
        por_grid = normal_transform(CF_grid,CF_glob,por_mean,np.sqrt(CF_var),np.sqrt(por_var))
        CF_grid[CF_grid > 1.] = 1.
        CF_grid[CF_grid < 0.] = 0.
        por_grid[por_grid > 1.] = .99
        por_grid[por_grid < 0.] = 0.01    
        hk_grid[wel_cells] = np.max((hk_grid.max(),200))
        # np.save(m.MC_file.parent.joinpath('{}_hk.npy'.format(ts)),hk_grid)
    else:
        pass


    # Write river data--take SSM data from WEL!!
    riv_grad = .0005
    riv_data, ssm_data = write_river_data(
        riv_loc, riv_stg, riv_cond, riv_grad, kper_even, ssm_data)
    ipakcb = 53
    icbund = np.ones((nlay, nrow, ncol), dtype=np.int)
    icbund[np.where(m.bas6.ibound.array == -1)] = -1
    timprs = np.round(np.linspace(1, np.sum(perlen), 20), decimals=0)
    oc_data = {}
    for kper in range(nper):
    	if kper % 5==0:
    		oc_data[(kper, 0)] = ['save head', 'save budget']

    flopy.modflow.ModflowBas(m, m.bas6.ibound.array, strt=strt)
    flopy.modflow.ModflowDis(m, nlay, nrow, ncol, nper=nper, delr=delr,
                               delc=delc,
                               laycbd=0, top=henry_top,
                               botm=henry_botm, perlen=perlen, nstp=nstp,
                               steady=steady, itmuni=itmuni, lenuni=lenuni,
                               tsmult=tsmult)


    flopy.modflow.ModflowGhb(m, stress_period_data=ghb_data)
    flopy.modflow.ModflowWel(m, stress_period_data=wel_data, ipakcb=ipakcb)
    flopy.modflow.ModflowRch(m, rech=rech_data)
    flopy.modflow.ModflowRiv(m, stress_period_data=riv_data)
	# Add LPF package to the MODFLOW model
    flopy.modflow.ModflowLpf(m, hk=hk_grid, vka=vka, ipakcb=ipakcb, laytyp=1,laywet=1,ss=ss,sy=sy)
	# Add PCG Package to the MODFLOW model
    flopy.modflow.ModflowPcg(m, hclose=1.e-8)
	# Add OC package to the MODFLOW model
    flopy.modflow.ModflowOc(m,stress_period_data=oc_data,compact=True)
    # Create the basic MT3DMS model structure
    flopy.mt3d.Mt3dBtn(m,
					laycon=m.lpf.laytyp, htop=henry_top,
					dz=m.dis.thickness.get_value(), prsity=por_grid, icbund=icbund,
					sconc=sconc, nprs=1, timprs=timprs)
    flopy.mt3d.Mt3dAdv(m, mixelm=-1)
    flopy.mt3d.Mt3dDsp(m, al=al, dmcoef=dmcoef)
    flopy.mt3d.Mt3dGcg(m, iter1=50, mxiter=1, isolve=1, cclose=1e-5)
    flopy.mt3d.Mt3dSsm(m, stress_period_data=ssm_data)

    #vdf = flopy.seawat.SeawatVdf(m, iwtable=0, densemin=0, densemax=0,denseref=1000., denseslp=0.7143, firstdt=1e-3)
    flopy.seawat.SeawatVdf(m, mtdnconc=1, mfnadvfd=1, nswtcpl=0, iwtable=1,
                                 densemin=0., densemax=0., denseslp=denseslp, denseref=densefresh)

    # Write input
    m.write_input()

    # Try to delete the output files, to prevent accidental use of older files
    flist = [os.path.join(model_ws, 'MT3D.CNF'),
             os.path.join(model_ws, 'MT3D001.MAS'),
             os.path.join(model_ws, modelname + '.hds'),
             os.path.join(model_ws, 'MT3D001.UCN'),
             os.path.join(model_ws, 'MT3D001.UCN'),
             os.path.join(model_ws, modelname + '.cbc')]
    for f in flist:
        try:
            os.remove(f)
        except:
            pass

    # Plot model? 
    if plotyn:
        m.plot_hk_ibound(rowslice=farm_orig[0][0],gridon=True)
        
    # Run model
    if runyn:
        v = m.run_model(silent=silent, report=True)
        for idx in range(-3, 0):  # Report
            print(v[1][idx])

        # Record success/failure and store data
        varlist['success'][it] = v[0]

        if v[0] is False:
            pass
        else:
            # Record final salinity as .npy, also move full CBC and UCN files
            # to expt folder
            fname = os.path.join(m.model_ws, 'MT3D001.UCN')
            totim = flopy.utils.binaryfile.UcnFile(fname).get_times()[-1]
            conc_fname = 'conc{}_{}_totim{}.UCN'.format(
                it, ts, str(int(totim)))
            _ = record_salinity(
                m, ts_hms=ts, fname_write=m.MC_file.parent.joinpath(conc_fname))
            utils.copy_rename(os.path.join(m.model_ws, 'MT3D001.UCN'),
                        m.MC_file.parent.joinpath(conc_fname).as_posix())
    if pooling:
        try:
            # [print(p) for p in tmp.iterdir() if (p.suffix is not '.UCN')]
            [p.unlink() for p in tmp.iterdir() if (p.suffix not in ('.UCN','.list'))]
            # shutil.rmtree(tmp.as_posix())
            # tmp.rmdir()
        except:
            print('didnt work!')
            pass
        m.model_ws = model_ws_orig
        print('resetting ws:',m.model_ws)

        if output is None:
            return (it,varlist['success'][it])
        else:
            output.put((it,varlist['success'][it]))
            # results.append((it,varlist['success'][it]))
            return
    else:
        utils.save_obj(m.MC_file.parent, (it,varlist['success'][it]), 'success{}'.format(it))
        [p.unlink() for p in model_ws.iterdir() if (p.suffix not in ('.UCN','.list'))]
        return m,varlist


def make_varlist_array(varlist,nwel=4):
    flag=0
    i=0
    for k,v in varlist.items():
        if flag==0:
            varlist_arr= np.zeros((len(varlist)+nwel,len(v)),dtype=np.float)
            flag=1
        if k is 'wel':
            for j in range(nwel):
                varlist_arr[i,:] = v[j,:]
                i+=1
        elif k is 'vario_type':
            varlist_arr[i,:] = [0 if model is 'Gaussian' else 1 for model in v]
            i+=1
        else:
            varlist_arr[i,:] = np.asarray(v,dtype=np.float)
            i+=1
    return varlist_arr
#%%






if __name__ == '__main__':
    runyn=True
    pooling=False
    # output = mp.Queue()

    m, varlist = update_run_model(varlist,it,runyn=runyn,
                                 start_basecase=True,silent=False,
                                 pooling=pooling,f_basecase=fpath_basecase)












