import os
from pathlib import Path,PureWindowsPath
import numpy as np
import flopy
from flopy.seawat import Seawat
import utils

#Subclass of flopy.seawat.Seawat
#    Carries relevant info about the ocean boundary and other info
class ModelSGD(Seawat):
    def __init__(self,ocean_arr = None,storage_dict=None, ocean_col=[30,69],ocean_bool=None,
                 MC_file=None,inputParams=None):
        super(Seawat, self).__init__()  #inherit Seawat properties

        #Set other properties
        self.ocean_bool = ocean_bool
        self.ocean_arr = ocean_arr
        self.storage_dict = storage_dict
        self.ocean_col = ocean_col
        self.MC_file = MC_file
        self.inputParams = inputParams
        print("made an SGD model!")
    #Take a Seawat object and turn it into an SGD object
    @classmethod
    def Seawat2SGD(self, objSeawat):
        objSeawat.load
        objSeawat.__class__ = ModelSGD
        return
    def set_ocean_arr(self,arr):
        self.ocean_arr = arr
        return
    def set_storage_dict(self,storage_dict):
        self.storage_dict = storage_dict
        return
    def write_output(self,fname='flux.smp'):
        #Get flux at ocean boundary
        d = utils.read_ref()
        if 'ocean_bool' in d:
            ocean_bool = np.load(d['ocean_bool'])
        elif self.ocean_arr is not None:
            ocean_bool = self.ocean_arr
        else:
            pass
        ocean_outflow = utils.get_ocean_outflow_chd(self,ocean_bool)[ocean_bool]
        ocean_sub = np.where(ocean_bool) #tuple of arrays giving indicies of ocean
        print('ocean_outflow: ',np.shape(ocean_outflow))
        print('ocean_bool: ',np.shape(ocean_bool))
        #Print out coordinates and flux to text file
        fout= open(os.path.join(self.model_ws,fname),"w")
        fout.write('Values are zero-based \n')
        fout.write('{:14s} {:4s} {:4s} {:4s} \n'.format("flux", "lay","row", "col"))
        for i in range(ocean_outflow.size):
            fout.write('{:14.4e} {:4d} {:4d} {:4d}\n'.format(ocean_outflow[i],ocean_sub[0]
                       [i],ocean_sub[1][i],ocean_sub[2][i]))
        fout.close()
        print('output FILE WRITTEN: ' + os.path.join(self.model_ws, fname))
        return

    #Copy the output file to create a synthetic observation file
    def copy_output2obs(self,suffix='.smp'):
        import shutil
        def get_fname(model_ws,ext):
            fname = [os.path.join(model_ws,f) for f in os.listdir(model_ws) if f.endswith(ext)][0];
            return fname
        f_output = get_fname(self.model_ws,suffix)
        f_obs = f_output[:-4] + '.obs'
        shutil.copy2(f_output,f_obs)
        print('observation file copied from: ' + f_output + '\nTo: ' + f_obs)

    #Write instruction file
    def write_ins(self,fname = 'flux.ins',obs_name = 'flux'):
        #Get flux at ocean boundary so you know how many lines to write
        d = utils.read_ref()
        if 'ocean_bool' in d:
            ocean_bool = np.load(d['ocean_bool'])
        elif self.ocean_arr is not None:
            ocean_bool = self.ocean_arr
        else:
            pass
        ocean_outflow = utils.get_ocean_outflow_chd(self,ocean_bool)[ocean_bool]

        #Write an instruction file
        finst = open(Path(os.path.join(self.model_ws,fname)).as_posix(),"w")
        finst.write('pif #\n')
        finst.write('#flux#\n')
        for i in range(ocean_outflow.size):
             finst.write('l1 w !{:s}!\n'.format(obs_name + '_' + str(i)))
        finst.close()
        print('.ins FILE WRITTEN: ' + Path(os.path.join(self.model_ws, fname)).as_posix())
        nobs = len(ocean_outflow)
        ins_data = [obs_name,nobs,ocean_outflow]
        return ins_data

    #Make template file
    def write_tpl(self,zonearray=None, parzones=None,lbound=0.001,
                  ubound=1000.,transform='log'):
        mfpackage = 'lpf'
        partype = 'hk'
        if zonearray == None:
            zonearray = np.ones((self.nlay, self.nrow, self.ncol), dtype=int)
            zonearray[2] = 2 #call layer 3 zone #2
        if parzones == None:
            parzones = [2]  #make zone #2 the zone to be parameterized
        lpf = self.get_package(mfpackage)
        parvals = [np.mean(lpf.hk.array)]

        plist = flopy.pest.zonearray2params(mfpackage, partype, parzones, lbound,
                                              ubound, parvals, transform, zonearray)
        tpl_data = [mfpackage, partype, parzones, lbound,
                                              ubound, parvals, transform, zonearray]
        # Write the template file
        tw = flopy.pest.templatewriter.TemplateWriter(self, plist)
        tw.write_template()
        print('.tpl FILE WRITTEN: ' + os.path.join(self.model_ws, self.name
                                                   +'.' + mfpackage + '.tpl'))
        npar = len(parzones)
        return tpl_data

    def write_pst(self,tpl_data,ins_data):
        import pandas

        fname = self.name + '.pst'
        f = open(os.path.join(self.model_ws,fname),"w")
        f.write('pcf\n')

        #Control data:
        '''
        RSTFLE PESTMODE
        NPAR NOBS NPARGP NPRIOR NOBSGP [MAXCOMPDIM] [DERZEROLIM]
        NTPLFLE NINSFLE PRECIS DPOINT [NUMCOM JACFILE MESSFILE] [OBSREREF]
        RLAMBDA1 RLAMFAC PHIRATSUF PHIREDLAM NUMLAM [JACUPDATE] [LAMFORGIVE] [DERFORGIVE]
        RELPARMAX FACPARMAX FACORIG [IBOUNDSTICK UPVECBEND] [ABSPARMAX]
        PHIREDSWH [NOPTSWITCH] [SPLITSWH] [DOAUI] [DOSENREUSE] [BOUNDSCALE]
        NOPTMAX PHIREDSTP NPHISTP NPHINORED RELPARSTP NRELPAR [PHISTOPTHRESH] [LASTRUN] [PHIABANDON]
        ICOV ICOR IEIG [IRES] [JCOSAVE] [VERBOSEREC] [JCOSAVEITN] [REISAVEITN] [PARSAVEITN] [PARSAVERUN]
        '''
        npar = len(tpl_data[2]) #length of parzones
        nobs = ins_data[1]
        npargp = 1 #number of param groups
        nprior = 0 #number of articles of prior info
        nobsgp = 1 #number of obs groups

        ntplfle = 1 #num tpl files
        ninsfle = 1 #num ins files
        precis = 'double'
        dpoint = 'point'

        rlambda1 = 10.0
        rlamfac = -3.0
        phiratsuf = 0.3
        phiredlam = 0.03
        numlam = 10

            #im not writing any more of these variables for now...
        f.write('* control data\n')
        f.write('restart estimation\n')
        f.write('{:d} {:d} {:d} {:d} {:d}\n'
                .format(npar,nobs,npargp,nprior,nobsgp))
        f.write('{:d} {:d} {:s} {:s}\n'
                .format(ntplfle,ninsfle,precis,dpoint))
        f.write('{:f} {:f} {:f} {:f} {:d}\n'
                .format(rlambda1,rlamfac,phiratsuf,phiredlam,numlam))
        f.write('10.0  10.0  0.001\n')
        f.write('0.1\n')
        f.write('30  0.005  4  4  0.005  4  1E-5\n')
        f.write('1  1  1\n')


        #LSQR
        '''
        LSQRMODE
        LSQR_ATOL LSQR_BTOL LSQR_CONLIM LSQR_ITNLIM
        LSQRWRITE
        '''
        #f.write('* lsqr\n')

        #parameter groups
        '''
        PARGPNME INCTYP DERINC DERINCLB FORCEN DERINCMUL DERMTHD [SPLITTHRESH SPLITRELDIFF SPLITACTION]
        '''
        f.write('* parameter groups\n')
        f.write('hk relative 0.01 0.0 switch 2.0 parabolic\n')

        #parameter data
        '''
        PARNME PARTRANS PARCHGLIM PARVAL1 PARLBND PARUBND PARGP SCALE OFFSET DERCOM
        '''
        parname = tpl_data[1] + '_' + str(tpl_data[2][0])
        partrans = 'log'
        parchglim = 'factor'
        parval1 = tpl_data[5][0]
        parlbnd = tpl_data[3]
        parubnd = tpl_data[4]
        parg = 'hk'
        scal = 1.0
        offset = 0.0
        dercom = 1

        f.write('* parameter data\n')
        f.write('{:s} {:s} {:s} {:f} {:e} {:e} {:s} {:f} {:f} {:d}\n'.
               format(parname,partrans,parchglim,parval1,parlbnd,parubnd,parg,scal,offset,dercom))

        #observation groups
        '''
        OBGNME [GTARG] [COVFLE]
        '''
        obgnme = 'heads'
        f.write('* observation groups\n')
        f.write('{:s}\n'.format(obgnme))

        #observation data
        def get_fname(model_ws,ext):
            fname = [os.path.join(model_ws,f) for f in os.listdir(model_ws) if f.endswith(ext)][0];
            return fname

        #"True" observations: should be same format as the .ins file and the .pts file
        fname_obs = get_fname(self.model_ws,'.obs')
        obs_array = pandas.read_csv(fname_obs,delim_whitespace=True,skiprows=1)
        obsval = obs_array.loc[:,'flux']
        weight = 10

        f.write('* observation data\n')
        for i in range(nobs):
            obsname = ins_data[0] + '_' + str(i)
            f.write('{:s} {:f} {:f} {:s}\n'.format(obsname,obsval[i],weight,obgnme))

        #model command line
        f.write('* model command line\n')
        f.write('SGD_runmodel.bat\n')

        #model input/output
        '''
        TEMPFLE INFLE
        INSFLE OUTFLE
        '''


        tempfle = get_fname(self.model_ws,'.tpl')
        infle = get_fname(self.model_ws,'.' + tpl_data[0])
        insfle = get_fname(self.model_ws,'.ins')
        outfle = get_fname(self.model_ws,'.smp')

        f.write('* model input/output\n')
        f.write('{:s} {:s}\n'.format(tempfle,infle))
        f.write('{:s} {:s}\n'.format(insfle,outfle))

        #DONE
        f.close()
        print('.pst FILE WRITTEN: ' + os.path.join(self.model_ws, fname))
        return

    def write_ref_file(self,d=None,ocean_bool_npy=None):
        from pathlib import Path
        if d!=None:
            pass
        elif d==None and self.storage_dict!=None:
            d = self.storage_dict
        elif d==None and self.storage_dict==None:
            d = {'model_ws':Path(self.model_ws).as_posix(),
            'modelname': self.name,
            'ocean_col': self.storage_dict['ocean_col'],
            }
        else:
            pass
        if ocean_bool_npy is None:
            ocean_bool_npy = str(Path(os.path.abspath(os.path.join(self.model_ws,'..','..','ocean_bool.npy'))).as_posix())
        d['ocean_bool'] = ocean_bool_npy
        try:
            os.remove(d['ocean_bool'])
        except:
            pass
        np.save(d['ocean_bool'], self.ocean_bool)
        #write the ocean_bool csv
        fname = Path(os.path.abspath(os.path.join(self.model_ws
                                                  ,'..','..','ref_file.txt'))).as_posix()
        fo = open(str(fname), "w")
        for k, v in d.items():
            fo.write('<<<' + str(k) + '>>>'+ str(v) + '\n')
        fo.close()
        print('reference FILE WRITTEN: ' + fname)
        return

    def plot_hk_ibound(self,rowslice=0,gridon=1,printyn=0,dpi=200):
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import numpy as np
        hk = self.get_package('LPF').hk.array

        # Make plot of the grid
        '''
        f = plt.figure(figsize=(6, 2))
        plt.clf()
        ax = f.add_subplot(1, 1, 1)
        '''
        f, ax = plt.subplots(1, figsize=(6, 2))
        plt.tight_layout()

        mm = flopy.plot.ModelCrossSection(ax=ax, model=self, line={'row':rowslice});
        hkpatchcollection = mm.plot_array(hk, norm=matplotlib.colors.LogNorm(),
                                          vmin=np.min(hk), vmax=np.max(hk),cmap='Greys_r');
        if gridon==1:
            mm.plot_grid();
        patchcollection = mm.plot_ibound(color_ch='orange');
        itype = flopy.mt3d.Mt3dSsm.itype_dict()
        for ftype in list(itype.keys()):
            try:
                mm.plot_bc(ftype=ftype)
            except:
                pass
        '''
        try:
            patchcollection3 = mm.plot_bc(ftype='WEL')
        except:
            pass
        try:
            patchcollection4 = mm.plot_bc(ftype='CHD')
        except:
            pass
        '''
        plt.ylabel('Elevation (m)')
        plt.xlabel('Distance (m)')
        plt.title('K-field & Boundary conditions');
        #align plots and set colorbar
        f.subplots_adjust(left=.1,right=0.88)

        if patchcollection:
            cb = plt.colorbar(patchcollection);
            cb.set_label('Boundary condition',rotation=90)
            cb.set_ticks((1.5,2.5))
            #cb.set_ticklabels(('No flow','Const head'))
            cb.ax.set_yticklabels(('No flow','Const head'),rotation=90)
        cbar_ax = f.add_axes([0.90, 0.12, 0.02, 0.7])
        cb2 = plt.colorbar(hkpatchcollection,cax=cbar_ax);
        cb2.set_label('Kh (m/d)', rotation=90)
        if printyn==1:
            fname=os.path.join(self.model_ws, self.name + '_BC_Hk.png')
            plt.savefig(fname,dpi=dpi)
            print('Saved to ' + fname)
        plt.show()
        return