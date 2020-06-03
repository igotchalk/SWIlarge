import matplotlib.pyplot as plt


def set_rc(SMALL_SIZE=12,MEDIUM_SIZE=12,BIGGER_SIZE=14):

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('font', family='Corbel')          # controls default text sizes

    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # sans-serif mathy-looking 
    plt.rc('mathtext',fontset = 'custom')
    plt.rc('mathtext',rm = 'Bitstream Vera Sans')
    plt.rc('mathtext',it = 'Bitstream Vera Sans:italic')
    plt.rc('mathtext',bf = 'Bitstream Vera Sans:bold')
    
    
    # More Latex-looking:
#     plt.rc('mathtext',fontset = 'stix')
#     plt.rc('font',family = 'DejaVu Sans')
    return