import sys
from pathlib import Path
import numpy as np


datadir = Path('/scratch/users/ianpg/SWIlarge/data')
workdir = Path('/scratch/users/ianpg/SWIlarge/work')
outputdir = workdir.joinpath('output')

simpegskytem_path = '../../kang-2019-3D-aem/codes'
if not simpegskytem_path in sys.path:
    sys.path.append(simpegskytem_path)
import simpegskytem

import numpy as np
from pathlib import Path

# waveform_dir = Path('aem_waveform_marina')
waveform_dir = datadir.joinpath('AEM','aem_waveform_marina')

### 304 Waveform ###
area = 337.04
unit_conversion = 1e-12

i_start_hm = 10
i_start_lm = 10

i_end_hm = -1
i_end_lm = -2

sl_hm = slice(i_start_hm, i_end_hm)
sl_lm = slice(i_start_lm, i_end_lm)

waveform_hm = np.loadtxt(waveform_dir.joinpath('hm_304.txt'))
waveform_lm = np.loadtxt(waveform_dir.joinpath('lm_304.txt'))
time_input_currents_HM = waveform_hm[:, 0]
input_currents_HM = waveform_hm[:, 1]
time_input_currents_LM = waveform_lm[:, 0]
input_currents_LM = waveform_lm[:, 1]

time_gates = np.loadtxt(waveform_dir.joinpath('time_gates.txt'))
GateTimeShift = -1.8E-06
MeaTimeDelay = 0.000E+00
NoGates = 28
t0_lm = waveform_lm[:, 0].max()
# times_LM = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_lm:] - t0_lm
times_LM = (time_gates[:NoGates, 0] +
                GateTimeShift + MeaTimeDelay)[sl_lm] - t0_lm

GateTimeShift = -1.4E-06
MeaTimeDelay = 6.000E-05
NoGates = 37
t0_hm = waveform_hm[:, 0].max()
# times_HM = (time_gates[:NoGates,0] + GateTimeShift + MeaTimeDelay)[i_start_hm:] - t0_hm
times_HM = (time_gates[:NoGates, 0] +
                GateTimeShift + MeaTimeDelay)[sl_hm] - t0_hm

# generate mesh
from SimPEG import Mesh, Utils
hx = np.ones(200) * 50
hy = np.ones(40) * 250
hz = np.ones(100) * 5
mesh_global = Mesh.TensorMesh([hx, hy, hz], x0=[-hx.sum()/2., -hy.sum()/2., -hz.sum()+10])

# generate survey
x_inds = np.argwhere(np.logical_and(mesh_global.vectorCCx > -4000, mesh_global.vectorCCx < 4000))[::20]
y_inds = np.argwhere(np.logical_and(mesh_global.vectorCCy > -4000, mesh_global.vectorCCy < 4000))[::20]
# x = mesh_global.vectorCCx[x_inds]
x = np.r_[-10000, 10000]
y = mesh_global.vectorCCy[y_inds]
src_locations = Utils.ndgrid(x, y, np.r_[30.])
rx_locations = Utils.ndgrid(x+13.25, y, np.r_[30.+2.])
topo = Utils.ndgrid(x, y, np.r_[0.])
source_area = 536.36
n_sounding = src_locations.shape[0]

actv = mesh_global.gridCC[:,2] < 0.
sigma = np.ones(mesh_global.nC) * 1e-8
sigma_background = 1./20.
sigma_target = 1./5.
thickness = np.array([50, 10], dtype=float)
depth = -np.cumsum(thickness)
inds = np.logical_and(mesh_global.gridCC[:,2]<depth[0], mesh_global.gridCC[:,2]>depth[1])
sigma[actv] = sigma_background
sigma[inds] = sigma_target
sigma[(mesh_global.gridCC[:,0]<0.) & actv] = sigma_background

from simpegskytem import (
    ProblemSkyTEM, GlobalAEMSurveyTD, 
    GlobalSkyTEM, get_skytem_survey
)
def get_skytem_survey(
    topo,
    src_locations,
    rx_locations,
    time,
    time_input_currents,
    input_currents,
    base_frequency=25,
    src_type="VMD",
    rx_type="dBzdt",    
    moment_type="dual",        
    time_dual_moment=None,
    time_input_currents_dual_moment=None,
    input_currents_dual_moment=None,
    base_frequency_dual_moment=210,
    wave_type="general",    
    field_type="secondary",
    
):
    
    n_sounding = src_locations.shape[0]    
    time_list = [time for i in range(n_sounding)]
    time_dual_moment_list = [time_dual_moment for i in range(n_sounding)]
    src_type_array = np.array([src_type], dtype=str).repeat(n_sounding)
    rx_type_array = np.array([rx_type], dtype=str).repeat(n_sounding)
    wave_type_array = np.array([wave_type], dtype=str).repeat(n_sounding)    
    field_type_array = np.array([field_type], dtype=str).repeat(n_sounding)  
    input_currents_list=[input_currents_HM for i in range(n_sounding)]
    time_input_currents_list=[time_input_currents_HM for i in range(n_sounding)]
    base_frequency_array = np.array([base_frequency]).repeat(n_sounding)
    input_currents_dual_moment_list =[input_currents_LM for i in range(n_sounding)]
    time_input_currents_dual_moment_list =[time_input_currents_LM for i in range(n_sounding)]
    base_frequency_dual_moment_list = np.array([base_frequency_dual_moment]).repeat(n_sounding)
    moment_type_array = np.array([moment_type], dtype=str).repeat(n_sounding)    
    
    survey = GlobalAEMSurveyTD(
        topo = topo,
        src_locations = src_locations,
        rx_locations = rx_locations,
        src_type = src_type_array,
        rx_type = rx_type_array,
        field_type = field_type,
        time = time_list,
        wave_type = wave_type_array,
        moment_type = moment_type_array,
        time_input_currents = time_input_currents_list,
        input_currents = input_currents_list,
        base_frequency = base_frequency_array,
        time_dual_moment = time_dual_moment_list,
        time_input_currents_dual_moment = time_input_currents_dual_moment_list,
        input_currents_dual_moment = input_currents_dual_moment_list,
        base_frequency_dual_moment = base_frequency_dual_moment_list,
    )    
    
    return survey
    
survey = get_skytem_survey(
    topo,
    src_locations,
    rx_locations,
    times_HM,
    time_input_currents_HM,
    input_currents_HM,
    25.,
    src_type="VMD",
    rx_type="dBzdt",    
    moment_type="dual",        
    time_dual_moment=times_LM,
    time_input_currents_dual_moment=time_input_currents_LM,
    input_currents_dual_moment=input_currents_LM,
    base_frequency_dual_moment=210.,
    wave_type="general",    
    field_type="secondary",    
)

simulation = GlobalSkyTEM(
    mesh_global, 
    sigma=sigma, 
    actv=actv,
    n_cpu=2,
#     parallel_option='dask',
#     work_dir='./tmp/'
)
simulation.pair(survey)

# write inputs to the disk
simulation.write_inputs_on_disk_pool()
# run 
data = simulation.forward(sigma)
simulation.clean_work_dir()