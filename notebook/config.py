# setup absolute paths to the executables based on the OS

import sys
import os
from pathlib import Path

print('system: {}'.format(sys.platform.lower()))
if sys.platform.lower() == 'darwin':
    bindir = Path('/usr/local/bin/')
    exepth = bindir.joinpath('MODFLOW')
    exeext = ''
elif sys.platform.lower() == 'linux':
    bindir = '/home/groups/rknight'
    exepth = bindir
    exeext = ''
elif 'win' in sys.platform.lower():
    print('Need to enter path for windows machine!!')
    # bindir = os.path.join(r'E:\Projects\DelawareSGD','bin')
    # exeext = '.exe'
    # winarch = 'win64'  #change to 'win32' for a 32-bit system
    # exepth = os.path.join(bindir, winarch)
else:
    raise Exception('Could not find binaries for {}'.format(sys.platform))

mfexe = exepth.joinpath('mf2005{}'.format(exeext)).absolute().as_posix()
mpexe = exepth.joinpath('mp6{}'.format(exeext)).absolute().as_posix()
mtexe = exepth.joinpath('mt3dms{}'.format(exeext)).absolute().as_posix()
swexe = exepth.joinpath('swt_v4{}'.format(exeext)).absolute().as_posix()

mf6exe = exepth.joinpath('mf6{}'.format(exeext)).absolute().as_posix()
mf2000exe = exepth.joinpath('mf2000{}'.format(exeext)).absolute().as_posix()
mflgrexe = exepth.joinpath('mflgr{}'.format(exeext)).absolute().as_posix()
mfnwtexe = exepth.joinpath('mfnwt{}'.format(exeext)).absolute().as_posix()
mfusgexe = exepth.joinpath('mfusg{}'.format(exeext)).absolute().as_posix()
mt3dusgsexe = exepth.joinpath(
    'mt3dusgs{}'.format(exeext)).absolute().as_posix()
gridgenexe = exepth.joinpath('gridgen{}'.format(exeext)).absolute().as_posix()


exelist = [mfexe, mpexe, mtexe, swexe, mf6exe, mf2000exe, mflgrexe,
           mfnwtexe, mfusgexe, mt3dusgsexe, gridgenexe]
for e in exelist:
    if not os.path.isfile(e):
        pass
        #print('Executable file could not be found: {}'.format(e))
