
# coding: utf-8

# In[50]:
import sys
from pathlib import Path
import numpy as np
import local_utils
import flopy
from simulationFFT import *
import discretize as Mesh
#Imports from simpegskytem notebook SLO-simulation
from simpegskytem import ProblemSkyTEM, GlobalAEMSurveyTD, GlobalSkyTEM, get_skytem_survey
import simpegskytem
from simpegskytem import Utils


'''
Input params:
it: iteration
datadir: directory where all the .UCN files are stored
'''
it = int(sys.argv[1])
datadir = Path(sys.argv[2])

print('Loading ucn data...')
try:
    ucn_dict = local_utils.load_obj(datadir,'ucn_inds')
except:
    ucn_dict = local_utils.create_ucn_dict(datadir)


#Load conc_mat
fname_conc = datadir.joinpath(ucn_dict[it])
times_select = np.arange(360*5,7201,360*5,dtype=np.float)
times = times_select
ucnobj = flopy.utils.binaryfile.UcnFile(Path(fname_conc).as_posix())
conc_mats = [ucnobj.get_data(totim=tim) for tim in times_select]

#Load and update varlist 
print('Loading varlist...')
try:
    varlist = local_utils.load_obj(datadir,'varlist_final')
except:
    varlist = local_utils.load_obj(datadir,'varlist')
    success_mat,_ = local_utils.unpack_success_files(datadir.joinpath('success'))
    varlist['success'][success_mat[:,0]]=success_mat[:,1]
    local_utils.save_obj(datadir,varlist,'varlist_final')



#RP transforms 
def WS_sigma(sigma_f, por = 0.4, CEC=1,B0=4.5e-8, m=1.3):
    rho_grain = 2650*1000 #g/m^3
#     CEC = 1 #meq/g    1:smect,  .2:Ill,  .02-.09:Kaol
#     B0 = 4.78e-8  #m^2/(sV)
    F = por**(-m)
    Qv = rho_grain*((1-por)/por)*CEC
    B = B0*(1-.6*np.exp(-sigma_f/.013))
    sigma_b = 1/F*(sigma_f + B*Qv)
    return sigma_b,B,Qv


def HSU(conc_mat,CF_mat,mTDS=1.4200556641030946,bTDS=332.7093594248108,**kwargs):
    #kwargs fed to WS_sigma
    Cw = (mTDS*conc_mat*1000 + bTDS)/1e4
    sig_bs = WS_sigma(Cw,CEC=0,**kwargs)[0]
    sig_bc = WS_sigma(Cw,CEC=1,**kwargs)[0]
    return sig_bc*(1 - (3*(1-CF_mat)*(sig_bc-sig_bs))/(3*sig_bc - CF_mat*(sig_bc-sig_bs)))


def rock_physics(varlist,it,conc_mat):
    nlay,nrow,ncol = (26, 20, 100)
    dx,dy,dz = (30,30,3)
    Lx,Ly,Lz = (dx*ncol,dy*nrow,dz*nlay)

    CF_glob = varlist['CF_glob'][it]
    CF_var = varlist['CF_var'][it]
    corr_len = varlist['corr_len'][it]
    corr_len_yx = varlist['corr_len_yx'][it]
    corr_len_zx = varlist['corr_len_zx'][it]
    vario_type = varlist['vario_type'][it]
    lcol = int(corr_len/dx)
    llay = int(corr_len*corr_len_zx/dz)
    lrow = int(corr_len*corr_len_yx/dy)
    seed = varlist['seed'][it]
    por_mean = varlist['por_mean'][it]
    por_var  = varlist['por_var'][it]
    m = varlist['m'][it]
    cf_mat = simulFFT(nrow,nlay, ncol,
                     CF_glob,
                     CF_var,
                     vario_type,
                     lrow , llay, lcol,seed=seed)
    cf_mat[cf_mat > 1.] = 1.
    cf_mat[cf_mat < 0.] = 0.

    por_mat = simulFFT(nrow,nlay, ncol,
                     por_mean,
                     por_var,
                     vario_type,
                     lrow , llay, lcol,seed=seed)
    por_mat[por_mat > 1.] = 1.
    por_mat[por_mat < 0.] = 0.
    sigma_bulk = HSU(conc_mat,cf_mat,por=por_mat,m=m)
    return sigma_bulk


def create_mesh():

    nlay,nrow,ncol = (26, 20, 100)
    dx,dy,dz = (30,30,3)
    Lx,Ly,Lz = (dx*ncol,dy*nrow,dz*nlay)

    n_pad_ocean = 20
    n_pad_inland = 20
    n_pad_row = 20
    n_sounding_x = ncol
    n_sounding_y = nrow

    hx = np.ones(ncol+n_pad_ocean+n_pad_inland) * dx
    hy = np.ones(nrow + 2*n_pad_row) * dy
    hz = np.ones(120)*dz
    x0 = (-n_pad_inland*dx,-n_pad_row*dy,-300)
    return Mesh.TensorMesh([hx, hy, hz],x0=x0)

def create_model(sigma_bulk,mesh):
    rho_transf = 1/sigma_bulk
    n_pad_ocean = 20
    n_pad_inland = 20
    n_pad_row = 20

    # #Air padding

    #Row padding
    rho_transf = np.append(rho_transf,
                           np.tile(rho_transf[:,np.newaxis,-1,:],(1,n_pad_row,1)),
                           axis=1)
    rho_transf = np.append(np.tile(rho_transf[:,np.newaxis,-1,:],(1,n_pad_row,1)),
                           rho_transf,
                           axis=1)
    #Ocean padding
    rho_transf = np.append(rho_transf,
                               rho_transf[:,:,-1].min()*np.ones(
                                   (rho_transf.shape[0],rho_transf.shape[1],n_pad_inland)),axis=2)



    #Inland padding
    rho_transf = np.append(rho_transf[:,:,0].mean()*np.ones(
                               (rho_transf.shape[0],rho_transf.shape[1],n_pad_inland)),
                           rho_transf,
                           axis=2)


    rho_background = 20.
    rho_grid = np.ones(mesh.vnC,order='F') * rho_background
    inds_insert = np.logical_and(mesh.vectorCCz> -26*3, mesh.vectorCCz <0)
    rho_grid[:,:,inds_insert] = np.transpose(rho_transf,(2,1,0))[:,:,::-1]
    rho_grid = rho_grid.flatten(order='F')
    rho_grid[mesh.gridCC[:,2]>0.] = np.nan

    actv = np.ones_like(rho_grid,dtype=bool)
    actv[mesh.gridCC[:,2]>0.] = False
    return rho_grid, actv


def create_survey(mesh,pad_amt=0,iskip=2):
    #Set source and receiver locations
    nlay,nrow,ncol = (26, 20, 100)
    dx,dy,dz = (30,30,3)
    Lx,Ly,Lz = (dx*ncol,dy*nrow,dz*nlay)
    xmin, xmax = pad_amt, Lx-pad_amt
    ymin, ymax = pad_amt, Ly-pad_amt

    # generate survey
    x_inds = np.argwhere(np.logical_and(mesh.vectorCCx > xmin, mesh.vectorCCx < xmax))
    y_inds = np.argwhere(np.logical_and(mesh.vectorCCy > ymin, mesh.vectorCCy < ymax))
    x = mesh.vectorCCx[x_inds][::iskip]
    y = mesh.vectorCCy[y_inds][::iskip]
    # f_dem = NearestNDInterpolator(dem[:,:2], dem[:,2])
    xy = Utils.ndgrid(x, y)
    # z = f_dem(xy)

    #Source and receiver height
    z = np.zeros(len(xy))
    src_height = 30.
    src_locations = np.c_[xy, z + src_height]
    rx_locations = np.c_[xy[:,0]+13.25, xy[:,1], z+2.+ src_height]
    topo = np.c_[xy, z]
    n_sounding = src_locations.shape[0]
    print(n_sounding)

    unit_conversion = 1e-12

    i_start_hm = 10
    i_start_lm = 10

    waveform_hm_312 = np.loadtxt('../aem-waveform/waveform_hm_312.txt')
    waveform_lm_312 = np.loadtxt('../aem-waveform/waveform_lm_312.txt')
    time_input_currents_hm_312 = waveform_hm_312[:,0] 
    input_currents_hm_312 = waveform_hm_312[:,1]
    time_input_currents_lm_312 = waveform_lm_312[:,0] 
    input_currents_lm_312 = waveform_lm_312[:,1]

    time_gates = np.loadtxt(('../aem-waveform/time_gates'))
    GateTimeShift=-2.09E-06
    MeaTimeDelay=0.000E+00
    NoGates=28
    t0_lm_312 = waveform_lm_312[:,0].max()
    times_lm_312 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_lm:] - t0_lm_312

    GateTimeShift=-1.5E-06
    MeaTimeDelay=3.500E-04
    NoGates=37
    t0_hm_312 = waveform_hm_312[:,0].max()
    times_hm_312 = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_hm:] - t0_hm_312    

    survey = get_skytem_survey(
        topo,
        src_locations,
        rx_locations,
        times_hm_312,
        time_input_currents_hm_312,
        input_currents_hm_312,
        base_frequency=25.,
        src_type="VMD",
        rx_type="dBzdt",    
        moment_type="dual",        
        time_dual_moment=times_lm_312,
        time_input_currents_dual_moment=time_input_currents_lm_312,
        input_currents_dual_moment=input_currents_lm_312,
        base_frequency_dual_moment=210.,
        wave_type="general",    
        field_type="secondary",    
    )
    return survey

def create_simulation(it,time,rho_grid,actv,mesh,survey):
    simulation_workdir = datadir.joinpath('./tmp{}_time{}'.format(it,int(time)))
    try:
        simulation_workdir.mkdir()
    except:
        pass

    simulation = GlobalSkyTEM(
        mesh, 
        sigma=1./rho_grid, 
        actv=actv, 
        parallel_option='multiprocess',
        n_cpu=8,
        work_dir=simulation_workdir.as_posix()
    )
    simulation.pair(survey)
    return simulation

def run_collect_data(simulation,rho_grid):
    print('writing inputs...\n')
    simulation.write_inputs_on_disk_pool()
    print('running simulation...\n')
    data = simulation.forward(1./rho_grid)
    print('finished simulation!')
    # DATA = data.reshape((times_hm_312.size+times_lm_312.size, n_sounding), order='F')
    # DATA_HM = -DATA[:times_hm_312.size,:]
    # DATA_LM = -DATA[times_hm_312.size:,:]
    return data


def main(conc_mats):
    data= []
    mesh = create_mesh()
    survey = create_survey(mesh,iskip=4)
    for tim,conc_mat in zip(times_select,conc_mats):
        try:
            survey.unpair()
        except:
            pass
        sigma_bulk = rock_physics(varlist,it,conc_mat)
        rho_grid,actv = create_model(sigma_bulk, mesh)
        simulation = create_simulation(it,tim,rho_grid,actv,mesh,survey)
        data.append(run_collect_data(simulation, rho_grid))
        try:
            survey.unpair()
        except:
            pass
    local_utils.save_obj(datadir,data,'aem_data_it{}'.format(it))
    print('saved to file aem_data_it{}'.format(it))
    return

if __name__ == '__main__':
    main(conc_mats)
