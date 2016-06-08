# JRK 8/12/13: maketables now determines the mode with the greatest
#  total likelihood and outputs 3 additional columns: its mean, sigma, and max.
# JRK 8/14/13: maketables and plotmarginal2 now use meas['addmass'] to 
#  convert from log(BACD/dv) to log(mass/dv), meas['addmass'] is now part of the 
#  measdata pickle.
# JRK 8/22/13: Fixed plotconditional2 and plotsledto work properly for 1 component modeling.
# JRK 10/30/13: In plot_marginal, allow overplotting of a different model for comparison,
#  parameterized by dists2 and cube2.
# JRK 7/28/15: Do not use the multiplot.multipanel package anymore for 
#  plotmarginal, plotmarginal2, and plotmarginalsled. Instead use subplots in matplotlib.
#  Still used in plotconditional and plotconditional2.
# JRK 8/13/15: Redid plotconditional and plotconditional2 to not use multiplot either; 
#  now it is not used at all. Also added diagonal lines for pressure and beam averaged
#  column density (BACD) on the tkin vs. h2den and cdmol vs. ff plots, respectively.
# JRK 8/28/15: get_dists: reduced distributions.pkl file size by:
#   1) do not replicate "0" and "all" distributions if only 1 mode.
#   2) do not calculate conditional probabilities between line flux likelihood and 
#       any other parameter.
#   Also creates a distributions1d.pkl file, which is < 1MB. Not used in these routines,
#   but can be used to quickly and easily access many 1D distributions.
#   plot_sled: added option legend=(True or False) and newplot=(True or False).
#     By default, this is used as newplot=True in pymultinest_analyze.py.
#     The false option allows one to insert the SLED on an existing plot object 
#     (e.g. for looping over many results and creating postage stamp SLED plots).
# JRK 9/16/15: Use return_dict=True in the plotsled call to pyradex, in order to 
#    use consistent keys with all other places in the code.
# JRK 9/29/15: In plot_sled, if lum=True, will now use meas['head']['dl'] instead of 
#    dist.Distance(z=meas2['head']['z']).Mpc. You MUST RERUN measdata_pickle in
#    your directory if your measdata.pkl file was created before today, it will be missing
#    that number in the header!
# JRK 12/16/15: Fixed bug introduced by 8/28 change - 'all' was a pointer to '0' in the 
#    case of one mode,  which messes up the normalization that is done in pyradexnest_analyze.py.
#    Now use copy.deepcopy(dists[0]).
#    Also added fix_flux_modemean, see its description for more info.
# JRK 12/18/15: fix_flux_modemean now recalculates all fluxes 'modemean' and 'modesigma'
# JRK 1/21/16: Added plot definitions for multiple molecule case.
#    Eventually can make all compatible with each other.
#    Also, in all cases, plotinds[2] are the secondary parameter ratios. Blank for 1 component.
#    This means plotinds[3] are the flux likelihoods, no matter how many components.
# JRK 1/22/16: Two new inputs for plotmarginalsled, useind=3 and mol='' is default behavior for plotting the 
#  primary molecule. Changing useind to 5, 6, etc. will plot the likelihoods for secondary molecules.
#  Putting a string for mol will add the molecule name to the filename.
# JRK 5/26/16: plotmarginalxmol had error attempting to index axes array if 
#   it was not in fact an array (just one plot).  plotmarginal, don't include xmol 
#   in calculation of number of dimensions (keep 2x2). plotmarginal2, correctly identify
#   the "nicenames" index for plot x-axis labels.
# JRK 6/2/16: Allow fix_flux_modemean to operate on secondary molecules as well, using useind
#   keyword. Fixed fignum error for plotmarginalxmol. Fixed "nicenames" indexing errors
#   for marginalized2 and marginalizedsled. Added "nbyn", short function that calculates
#   the best subplot arrangement for a given number of parameters. Add molecule abundance
#   to secondary molecule column density when RADEX is called again for tau and tex.
#   If luminosity was not calculated, annotate that and remote axis labels for that 
#   plot in plotconditional2 and plotmarginal2.

import numpy as np
import astropy.units as units
#from multiplot import multipanel
import matplotlib.pyplot as plt
import pyradex as pyradex
import matplotlib.mlab as mlab
import copy

def define_plotting_multimol(n_comp,n_mol,n_dims,n_sec,n_params,sled_to_j,lw,compare=False):
   
    mult=np.ones(n_params,'i') # Multiplication is not used in RADEX modeling.
    add=np.zeros(n_params,'f') # Fill in later
    ccolor='b' if not compare else 'k'
    colors=np.array([ccolor for x in range(n_params)])
   
    if n_comp==1:
        parameters=["h2den1","tkin1","cdmol1","ff1"]
        for i in range(n_mol-1): parameters.append('xmol'+str(i+1))
        for i in ["lum1","press1","bacd1"]: parameters.append(i)
        if sled_to_j: 
            for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m) for i in range(1,sled_to_j+1)])
        add[[2,4+(n_mol-1),6+(n_mol-1)]]=lw
        sledcolors=['b'] if not compare else ['#6495ED']
        
        plotinds=[[0,1,2,3],[n_dims+i for i in range(3)],[]] # Primary 4 parameters, Secondary 3, none for ratio
        plotinds.append(range(n_dims+np.sum(n_sec),n_dims+np.sum(n_sec)+sled_to_j,1)) # Flux likelihoods, Main molecule
        plotinds.append([4+i for i in range(n_mol-1)]) # NEW, secondary molecule abundances
        # NEW, secondary molecule flux likelihoods
        if sled_to_j:
            for m in range(n_mol-1): 
                plotinds.append([n_dims+np.sum(n_sec)+(m+1)*sled_to_j+i for i in range(sled_to_j)])
        
    elif n_comp==2:
        parameters = ["h2den1","tkin1","cdmol1","ff1","h2den2","tkin2","cdmol2","ff2"]
        for i in range(n_mol-1): parameters.append('xmol'+str(i+1)+'c')
        for i in range(n_mol-1): parameters.append('xmol'+str(i+1)+'w')
        for i in ["lum1","press1","bacd1","lum2","press2","bacd2",
              "lumratio","pressratio","bacdratio"]: parameters.append(i)
        if sled_to_j:
            for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m)+"c" for i in range(1,sled_to_j+1)]) # Cold
            for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m)+"w" for i in range(1,sled_to_j+1)]) # Warm

        add[[2,6,8+(n_mol-1)*2,10+(n_mol-1)*2,11+(n_mol-1)*2,13+(n_mol-1)*2]]=lw
        
        # Change the colors for the warm component
        # But first, if compare, don't use blue or black.
        if compare: colors[:]='#6495ED'
        wcolor='r' if not compare else 'm'
        rcolor='k' if not compare else 'gray'
        colors[[4,5,6,7]]=wcolor  # Primary parameters
        colors[[n_comp*4+(n_mol-1)+i for i in range(n_mol-1)]]=wcolor # Xmol
        colors[[n_dims+i for i in range(3,6)]]=wcolor # Secondary parameters
        colors[[n_dims+i for i in range(6,9)]]=rcolor # Ratios
        colors[[n_dims+9+sled_to_j*n_mol+i for i in range(sled_to_j*n_mol)]]=wcolor # Flux likelihoods

        sledcolors=['b','r'] if not compare else ['#6495ED','m'] 
        
        plotinds=[[0,1,2,3,4,5,6,7],[n_dims+i for i in range(6)],[n_dims+i for i in range(7,10)]]
        # Main molecule, cold then warm all together
        moladd=np.concatenate([np.arange(sled_to_j),np.arange(sled_to_j)+sled_to_j*n_mol])
        plotinds.append([n_dims+np.sum(n_sec)+i for i in moladd])
        # NEW, secondary molecule abundances
        plotinds.append([8+i for i in range((n_mol-1)*2)])
        # NEW, secondary molecule flux likelihoods, each molecule includes cold then warm all together
        if sled_to_j:
            for m in range(n_mol-1):
               moladd=np.concatenate([np.arange(sled_to_j),np.arange(sled_to_j)+sled_to_j*n_mol])
               plotinds.append([n_dims+np.sum(n_sec)+(m+1)*sled_to_j+i for i in moladd])

    # Check that we have the right number of parameters!
    if n_params != len(parameters):
        raise Exception('Number of parameters calculation is off!')

    return [parameters,add,mult,colors,plotinds,sledcolors]

def define_plotting(n_dims,n_sec,sled_to_j,lw,compare=False):
    # Based on the number of parameters, return the information needed for plotting.
    # If this is a comparison, set compare=True, as this changes the colors to be used.
    
    n_comp=1 if n_dims==4 else 2 # Added 1/21 for compatibility with multi molecule.

    if n_comp==1:  # 1 component
        parameters = ["h2den1","tkin1","cdmol1","ff1","lum1","press1","bacd1"] 
        add=[0,0,lw,0, lw,0,lw,]
        mult=[1,1,1,1, 1,1,1] # Multiplication is NOT something used for CO RADEX modeling.
        colors=['b','b','b','b','b','b','b']
        if compare: colors2=['k','k','k','k','k','k','k']
        plotinds=[[0,1,2,3],[4,5,6],[]] # 1/21/16: add empty plotinds[2] for ratios
        sledcolors=['b']
        if compare: sledcolors=['#6495ED']
        # Add if we also have likelihoods CO fluxes, append the required information for the likelihood plot.
        for x in range(sled_to_j): 
            parameters.append('flux'+str(x+1))
            add.append(0)
            mult.append(1)
            colors.append('#6495ED') if compare else colors.append('k')
        plotinds.append(range(n_dims+np.sum(n_sec),n_dims+np.sum(n_sec)+sled_to_j,1))
    elif n_comp==2:  # 2 component
        parameters = ["h2den1","tkin1","cdmol1","ff1",
                      "h2den2","tkin2","cdmol2","ff2",
                      "lum1","press1","bacd1",
                      "lum2","press2","bacd2",
                      "lumratio","pressratio","bacdratio"] 
        add=[0,0,lw,0, 0,0,lw,0, lw,0,lw, lw,0,lw, 0,0,0]
        mult=[1,1,1,1, 1,1,1,1, 1,1,1, 1,1,1, 1,1,1]
        colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k']
        if compare: colors=['#6495ED','#6495ED','#6495ED','#6495ED','m','m','m','m', \
                    '#6495ED','#6495ED','#6495ED','m','m','m','gray','gray','gray']
        plotinds=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16]]
        sledcolors=['b','r']
        if compare: sledcolors=['#6495ED','m']
        # Add if we also have likelihoods CO fluxes, append the required information for the likelihood plot.
        for x in range(sled_to_j): 
            parameters.append('flux'+str(x+1)+'c')
            add.append(0)
            mult.append(1)
            colors.append('#6495ED') if compare else colors.append('b')
        for x in range(sled_to_j): 
            parameters.append('flux'+str(x+1)+'w')
            add.append(0)
            mult.append(1)
            colors.append('m') if compare else colors.append('r')      
        plotinds.append(range(n_dims+np.sum(n_sec),n_dims+np.sum(n_sec)+2*sled_to_j,1))
        
    return [parameters,add,mult,colors,plotinds,sledcolors]

def post_sep(post_file):
    # Where are the blank lines in the 1-post_separate.dat file?  Note starting at 0.
    f=open(post_file,"r")
    blank=[]
    for i,line in enumerate(f):
        if not line.strip(): blank.append(i)
    f.close()
    # We expect 0,1 to be blank always.  There will be 2x*nmodes blank.
    ind=range(2,len(blank)-1,2)
    breakind=[]
    for i in ind: breakind.append(blank[i]-(i))
    
    postdat=np.loadtxt(post_file)
    breakind.insert(0,0)
    breakind.append(len(postdat))
    nmodes=len(breakind)-1
    datsep={}
    for i in range(nmodes):datsep[i]=postdat[breakind[i]:breakind[i+1]]
    
    return datsep

def arrowloglen(ylim,top,frac=32.0):
    yr=np.log10(ylim)
    delta=(yr[1]-yr[0])/frac
    bottom=10.0**(np.log10(top)-delta)
    return bottom-top

def bin_results(stats,data,dim1,dim2=None,grid_points=40,
                marginalization_type='sum', only_interpolate=False,
                use_log_values=False):
    import itertools
    """
    See plot.py from pymultinest.
    """   
    n = grid_points 
    
    modes = stats['modes']
    min1 = min([mode['mean'][dim1] - 3*mode['sigma'][dim1] for mode in modes])
    max1 = max([mode['mean'][dim1] + 3*mode['sigma'][dim1] for mode in modes])
    if dim2 is not None:
        min2 = min([mode['mean'][dim2] - 3*mode['sigma'][dim2] for mode in modes])
        max2 = max([mode['mean'][dim2] + 3*mode['sigma'][dim2] for mode in modes])
    
    grid_x = np.mgrid[min1:max1:n*1j]
    if dim2 is not None:
        m = n
        grid_x, grid_y = np.mgrid[min1:max1:n*1j, min2:max2:n*1j]
        binsize2 = (max2 - min2) / m
    else:
        m = 1
        grid_x = np.mgrid[min1:max1:n*1j]
        grid_y = [0]
        
    binsize1 = (max1 - min1) / n
    
    dim1_column = data[:,2 + dim1]
    if dim2 is not None:
        dim2_column = data[:,2 + dim2]
        coords = np.array([dim1_column, dim2_column]).transpose()
    else:
        coords = dim1_column.transpose()
    values = data[:,0]
    if use_log_values:
        values = np.log(values)
    grid_z = np.zeros((n,m))
    minvalue = values.min()
    maxvalue = values.max()
    
    # for each grid item, find the matching points and put them in.
    for row, col in itertools.product(range(len(grid_x)), range(len(grid_y))):
        if dim2 is not None:
            xc = grid_x[row,col]
            here_x = np.abs(dim1_column - xc) < binsize1 / 2.
            yc = grid_y[row,col]
            here_y = np.abs(dim2_column - yc) < binsize2 / 2.
        else:
            xc = grid_x[row]
            here_x = np.abs(dim1_column - xc) < binsize1 / 2.
            here_y = True
        
        bin = values[np.logical_and(here_x, here_y)]
        if bin.size != 0:
            if marginalization_type == 'max':
                grid_z[row,col] = bin.max()
            elif marginalization_type == 'sum':
                grid_z[row,col] = bin.sum()
            elif marginalization_type == 'mean':
                grid_z[row,col] = bin.mean()
            elif marginalization_type == 'count':
                grid_z[row,col] = bin.size
            else:
                assert False, "marginalization_type should be mean, sum or max"
        else:
            grid_z[row,col] = minvalue

    #print 'maxima', values.max(), grid_z.max()
    # plot gridded data
    if only_interpolate:
    #   version A: interpolated -- may look weird because of the 
    #              loss of dimensions
        assert dim2 is not None
        grid_z = scipy.interpolate.griddata(coords, values, (grid_x, grid_y), method='cubic')
        
    return (grid_x,grid_y,grid_z)

def get_dists(distfile,s,datsep,maxcind,grid_points=40):
    # To reduce the size of the distributions files...
    #   1) If there is only one mode, do not replicate "0" and "all" when creating or saving.
    #      Unfortunately, if we want to keep each mode separate as well as the total, 
    #      we'll have larger files.
    #   2) Do not calculate conditional probabilities between flux likelihoods and 
    #      any other parameter likelihoods. For 2-component fits with 13 lines, 
    #      this was 85% of the distribution size. The index at which flux likelihoods
    #      start is now input as maxcind (maximum conditional index), n_dims + np.sum(n_sec)
    #      Update 5/26/16: 
    import os
    import cPickle as pickle
    n_params=len(s['modes'][0]['mean'])
    nmodes=len(s['modes'])
    # Check if we already have distributions.pkl file, and if one of the outputs in 
    #   chains has not changed since distribution.pkl's modified time.
    if os.path.exists(distfile) and os.path.exists('chains/1-.json') and os.path.getmtime(distfile) > os.path.getmtime('chains/1-.json'):
        try:
            dists=pickle.load(open(distfile,"rb"))
            redodists=False
        except: 
            redodists=True
    else:
        redodists=True
    if redodists:
        # We want 1D distributions for each parameter, 2D for each combination of parameters
       # The secondary paramters (luminosity, pressure, bacd) are included here.
        dists={}
        dists1d={}
        if nmodes==1: del datsep['all'] # Don't need to do twice.
        print datsep.keys()
        for key in datsep.keys():
            print 'Marginalizing...',key
            data=datsep[key]
            tdists={}
            tdists1d={}
            maxes=[]
            for i in range(n_params):
                x,y,z = bin_results(s,data,i,dim2=None,grid_points=grid_points,
                                    marginalization_type='sum', only_interpolate=False)
                z=z.reshape(grid_points)
                tdists[i]=np.vstack((x,z))
                tdists1d[i]=np.vstack((x,z))
                
                # If we are going to do normalization of multiple modes, we need to 
                # keep the "max" of each one easily accessible here.
                maxes.append(z.max())
                
                if i<maxcind:
                    for j in range(i):
                        x,y,z = bin_results(s,data,i,dim2=j,grid_points=grid_points,
                                        marginalization_type='sum', only_interpolate=False)
                        tdists[i,j]=np.dstack((x,y,z))
            tdists['max']=maxes
            tdists1d['max']=maxes
            dists[key]=tdists
            dists1d[key]=tdists1d
            
        pickle.dump(dists, open(distfile, "wb") )
        pickle.dump(dists1d, open(distfile.replace('.pkl','1d.pkl'), "wb") )
        print 'Saved distributions.pkl'    
        
    if nmodes==1: dists['all']=copy.deepcopy(dists[0]) # Redundancy used later... at least not on disk :-/
        
    return dists
    
def maketables(s,n_params,parameters,cube,add,mult,n_comp,title='results',addmass=0,n_dims=8):
    from astropy.table import Table, Column
    # Require n_comp
    
    # If there are multiple modes, determine the most likely one.
    localev=[]
    for j in s['modes']: 
        try:
            localev.append(j['local evidence'])
        except:
            localev.append(j[u'local log-evidence'])
    maxind=localev.index(max(localev))
       
    col1=[]
    col2=[]
    col3=[]
    col4=[]
    col5=[]
    col6=[]
    col7=[]
    for j in range(n_params):
        col1.append(s['marginals'][j]['median']*mult[j]+add[j])    # Marginalized Median
        col2.append(s['marginals'][j]['1sigma'][0]*mult[j]+add[j]) # -1Sigma Marginalized
        col3.append(s['marginals'][j]['1sigma'][1]*mult[j]+add[j]) # +1Sigma Marginalized
        col4.append(cube[j]*mult[j]+add[j])                                # Best Fit
        col5.append(s['modes'][maxind]['mean'][j]*mult[j]+add[j])       # Mode Mean
        col6.append(s['modes'][maxind]['sigma'][j]*mult[j]+0)      # Mode Sigma
        col7.append(s['modes'][maxind]['maximum'][j]*mult[j]+add[j])    # Mode Maximum
    
    # Append mass data
    if addmass !=0:  # No multplicatino here, just addition.
        bacdinds=[parameters.index('bacd1')]
        parameters.append('mass1')
        if n_comp==2: 
            bacdinds=[parameters.index('bacd1'),parameters.index('bacd2')]
            parameters.append('mass2')       
 
        for bacdind in bacdinds:
            col1.append(s['marginals'][bacdind]['median']     +add[bacdind]+addmass)
            col2.append(s['marginals'][bacdind]['1sigma'][0]  +add[bacdind]+addmass)
            col3.append(s['marginals'][bacdind]['1sigma'][1]  +add[bacdind]+addmass)
            col4.append(cube[bacdind]                         +add[bacdind]+addmass)
            col5.append(s['modes'][maxind]['mean'][bacdind]   +add[bacdind]+addmass)
            col6.append(s['modes'][maxind]['sigma'][bacdind]                       )
            col7.append(s['modes'][maxind]['maximum'][bacdind]+add[bacdind]+addmass)

    table=Table([parameters,col1,col2,col3,col4,col5,col6,col7],
        names=('Parameters','Median','-1 Sigma','+1 Sigma','Best Fit','Mode Mean','Mode Sigma','Mode Maximum'))
    # For the ascii file, use more digits.  And make the names be separated by spaces only.
    table.write("results_ascii.txt",format="ascii",names=('param','med','-1sig','+1sig','bestfit','modemean','modesigma','modemax'))

    # But for latex, just need 2 digits.  Well, for the fluxes, we'll want more, so add more...
    table['Median'].format='%.4f' # I am aware this is stupid.
    table['-1 Sigma'].format='%.4f'
    table['+1 Sigma'].format='%.4f'
    table['Best Fit'].format='%.4f'
    table['Mode Mean'].format='%.4f'
    table['Mode Sigma'].format='%.4f'
    table['Mode Maximum'].format='%.4f'
    
    table.pprint()
    table.write("results_latex.tex",format="latex",caption=title.replace('_',' '))
    
    return table
###############################################################################################

def plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,
                 xr=[[2,6.5],[0.5,3.5],[13,23],[-3,0]],title='',norm1=True,
                 modecolors=[[0,0],[0.1,0.9],[0.2,0.5],[0.3,0.7,1]],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=0,simplify=False,meas=0):
    import matplotlib.mlab as mlab
    #mp = multipanel(dims=(np.mod(n_dims,4)/2+2,2),figID=1,padding=(0,0.2),panel_size=(1,np.mod(n_dims,4)/2+1)) # 
    #if not simplify: mp.title(title, xy=(0.5, 0.97))
    nx=2
    ny=2 # np.mod(n_dims-(n_mol-1),4)/2+2 # Do not add plots for xmol; plotted in plotmarginalxmol
    f, axarr=plt.subplots(ny,nx,num=1,sharey=True,figsize=(4*nx,4*ny))
    f.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    f.subplots_adjust(wspace=0.1)
    if not simplify: axarr[0][0].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
    #gridinds=[2,3,0,1]
   #if n_dims==8: gridinds=[2,3,0,1,2,3,0,1]  # multiplot is now different indices...
    #f dists2:
    #    gridinds=[2,3,0,1]
    #    if n_dims2==8: gridinds=[2,3,0,1,2,3,0,1]  # multiplot is now different indices...
    
    for g,j in enumerate(plotinds[0]):   
        gridinds=np.floor((np.mod(g,4))/nx),np.mod(g,nx)
        if not dists2 and not simplify: # Don't do these if we are overplotting a 2nd dist, too confusing, or Simplify is set!
            for m,mode in enumerate(s['modes']):
                if colors[j]=='b': 
                    modecol=[modecolors[m][0],modecolors[m][1],1]
                elif colors[j]=='r':
                    modecol=[1,modecolors[m][1],modecolors[m][0]]
                elif colors[j]=='g': # Added JRK 1/23/14
                    modecol=[modecolors[m][0],1,modecolors[m][1]]
                elif colors[j]=='k': # Added JRK 5/17/14
                    modecol=[modecolors[m][0],modecolors[m][0],modecolors[m][0]]
                
                yplot=dists[m][j][1,:]
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =np.ravel(zip(yplot, yplot))
                
                #mp.grid[gridinds[g]].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 
                axarr[gridinds].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                ##print mode['maximum'][j]+add[j],mode['sigma'][j]
            
        yplot=dists['all'][j][1,:] 
        axarr[gridinds].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        #mp.grid[gridinds[g]].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
      
    # Comparison distributions overplotted.  
    if dists2:
        for g,j in enumerate(plotinds2[0]):
            gridinds=np.floor((np.mod(g,4))/nx),np.mod(g,nx)   
            yplot2=dists2['all'][j][1,:]
            axarr[gridinds].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
            axarr[gridinds].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')            
    
    # Ranges and labels
    #mp.fix_ticks()
    axarr[0][0].set_ylabel("Likelihood")
    axarr[1][0].set_ylabel("Likelihood")
    for i in range(4):
        gridinds=np.floor((i)/nx),np.mod(i,nx)   
        axarr[gridinds].set_xlabel(nicenames[i])  # x-axis labels
        axarr[gridinds].set_xlim(xr[i])           # x-axis ranges.
        if norm1: axarr[gridinds].set_ylim(0,1)   # y-axis ranges
    
    if not simplify:   
        axarr[0][0].annotate('ln(like):',xy=(0.8,0.9),xycoords='axes fraction')
        for m,mode in enumerate(s['modes']): 
            try:
                axarr[0][0].annotate('%.2f' % mode['local evidence'],xy=(0.8,0.8-0.1*m),xycoords='axes fraction',color=[modecolors[m][0],modecolors[m][1],1])
                #axarr[0][0].text(4,0.9-0.1*m,,color=[modecolors[m][0],modecolors[m][1],1])
            except:
                axarr[0][0].annotate('%.2f' % mode['local log-evidence'],xy=(0.8,0.8-0.1*m),xycoords='axes fraction',color=[modecolors[m][0],modecolors[m][1],1])
                #axarr[0][0].text(4,0.9-0.1*m,'%.2f' % mode[u'local log-evidence'],color=[modecolors[m][0],modecolors[m][1],1])

    plt.draw()
    plt.savefig('fig_marginalized.png')
    print 'Saved fig_marginalized.png'
    
def plotmarginalxmol(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,n_mol,nicenames,
                 colors,
                 xr=[[2,6.5],[0.5,3.5],[13,23],[-3,0]],title='',norm1=True,
                 modecolors=[[0,0],[0.1,0.9],[0.2,0.5],[0.3,0.7,1]],
                 dists2={},colors2=['g','m'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=0,n_mol2=0,simplify=False,meas=0):
    import matplotlib.mlab as mlab
    

    nx=n_mol-1
    ny=1
    plt.figure(num=6)
    plt.clf()
    f, axarr=plt.subplots(ny,nx,num=6,sharey=True,figsize=(4*nx,4*ny)) # Fixed 6/2/16, was num=1
    f.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    f.subplots_adjust(wspace=0.1)
    if not simplify: axarr[0][0].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
    for g,j in enumerate(plotinds[4]):   
        # If we only have 1 plot, we do NOT want gridinds.
        if nx>1: gridinds=np.mod(g,n_mol-1)
        if not dists2 and not simplify: # Don't do these if we are overplotting a 2nd dist, too confusing, or Simplify is set!
            for m,mode in enumerate(s['modes']):
                if colors[j]=='b': 
                    modecol=[modecolors[m][0],modecolors[m][1],1]
                elif colors[j]=='r':
                    modecol=[1,modecolors[m][1],modecolors[m][0]]
                elif colors[j]=='g': # Added JRK 1/23/14
                    modecol=[modecolors[m][0],1,modecolors[m][1]]
                elif colors[j]=='k': # Added JRK 5/17/14
                    modecol=[modecolors[m][0],modecolors[m][0],modecolors[m][0]]
                
                yplot=dists[m][j][1,:]
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =np.ravel(zip(yplot, yplot))
                
                #mp.grid[gridinds[g]].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 
                if nx==ny==1:
                    axarr.fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 
                    axarr.axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                    axarr.axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                    axarr.axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')                
                else:
                    axarr[g].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 
                    axarr[g].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                    axarr[g].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                    axarr[g].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                ##print mode['maximum'][j]+add[j],mode['sigma'][j]
            
        yplot=dists['all'][j][1,:] 
        if nx==ny==1:
            axarr.plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        else:
            axarr[gridinds].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
            #mp.grid[gridinds[g]].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
      
    # Comparison distributions overplotted.  
    if dists2:
        for g,j in enumerate(plotinds2[4]):
            gridinds=np.mod(g,n_mol2-1)
            yplot2=dists2['all'][j][1,:]
            if nx==ny==1:
                axarr.plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                axarr.axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')            
            else:
                axarr[gridinds].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                axarr[gridinds].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')            
    
    # Ranges and labels
    #mp.fix_ticks()
    if nx==ny==1:
        axarr.set_ylabel("Likelihood")
        axarr.set_xlabel(nicenames[4])  # x-axis labels
        axarr.set_xlim(xr[4])           # x-axis ranges.
        if norm1: axarr.set_ylim(0,1)   # y-axis ranges    
    else:
        for i in range(n_mol-1):
            axarr[i].set_ylabel("Likelihood")
            axarr[i].set_xlabel(nicenames[4+i])
            axarr[i].set_xlim(xr[4+i])
            if norm1: axarr[i].set_ylim(0,1)
        

    if not simplify and not nx==ny==1:
        axarr[0][0].annotate('ln(like):',xy=(0.8,0.9),xycoords='axes fraction')
        for m,mode in enumerate(s['modes']): 
            try:
                axarr[0][0].annotate('%.2f' % mode['local evidence'],xy=(0.8,0.8-0.1*m),xycoords='axes fraction',color=[modecolors[m][0],modecolors[m][1],1])
                #axarr[0][0].text(4,0.9-0.1*m,,color=[modecolors[m][0],modecolors[m][1],1])
            except:
                axarr[0][0].annotate('%.2f' % mode['local log-evidence'],xy=(0.8,0.8-0.1*m),xycoords='axes fraction',color=[modecolors[m][0],modecolors[m][1],1])
                #axarr[0][0].text(4,0.9-0.1*m,'%.2f' % mode[u'local log-evidence'],color=[modecolors[m][0],modecolors[m][1],1])

    plt.draw()
    plt.savefig('fig_marginalized_xmol.png')
    print 'Saved fig_marginalized_xmol.png'
    
def plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_comp,n_sec,n_dims,nicenames,n_mol,
                 xr=[[2,6.5],[0.5,3.5],[13,23],[-3,0]],title='',norm1=True,
                 modecolors=['g','m','y','c'],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],meas=0,
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=8,simplify=False):
    
    #mp=multipanel(dims=(1,3),figID=2,padding=(0,0),panel_size=(3,1)) # figID=2,
    #plt.clf()
    #mp=multipanel(dims=(1,3),figID=2,padding=(0,0),panel_size=(3,1)) # 
    #plt.subplots_adjust(bottom=0.12,left=0.06,right=0.98,top=0.88)
    #if not simplify: mp.title(title, xy=(0.5, 0.97))
    nx=3
    ny=1
    f, axarr=plt.subplots(ny,nx,num=2,sharey=True,figsize=(4*nx,4*ny))
    f.subplots_adjust(bottom=0.12,left=0.06,right=0.98,top=0.88)
    f.subplots_adjust(wspace=0.1)
    if not simplify: axarr[1].set_title(title)    
    
    
    for j in plotinds[1]:        
        # The filling was not adding up right; don't plot each individual mode.
        if not dists2 and not simplify: # Too confusing otherwise
            for m,mode in enumerate(s['modes']):
            #    yplot=dists[m][j][1,:]
            #    #mp.grid[np.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
            #    dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
            #    xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
            #    yy =np.ravel(zip(yplot, yplot))
            #    mp.grid[np.mod(j-n_dims,3)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
                
                axarr[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
                axarr[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],    
                       color=modecolors[m],label='Mode',linestyle='--')    
                axarr[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],
                       color=modecolors[m],label='Mode',linestyle='--')
            
        yplot=dists['all'][j][1,:]
        axarr[np.mod(j-n_dims,3)].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        axarr[np.mod(j-n_dims,3)].axvline(x=cube[j]*mult[j]+add[j],color=colors[j])
        if norm1: axarr[np.mod(j-n_dims,3)].set_ylim(0,1)   
        
        if j==n_dims:
            ax2a=axarr[np.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]*mult[j]+add[j]-np.log10(units.solLum.to('erg/s'))
            ax2a.plot(newx,yplot,'-',color=colors[j], drawstyle='steps')
            if np.max(dists['all'][j][1,:])==1: # If luminosity is not calculated, public radex.
                #axarr[np.mod(j-n_dims,3)].axis('off')
                axarr[np.mod(j-n_dims,3)].annotate('Luminosity Not Calculated',xy=(0.1,0.5),xycoords='axes fraction')
                [label.set_visible(False) for label in axarr[np.mod(j-n_dims,3)].get_xticklabels()] 
                [label.set_visible(False) for label in ax2a.get_xticklabels()] 
        
        if j==n_dims+2:
            ax2b=axarr[np.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]*mult[j]+add[j]+meas['addmass']
            ax2b.plot(newx,yplot,'-', color=colors[j], drawstyle='steps')
            
    if dists2:
        for j in plotinds2[1]:
            yplot2=dists2['all'][j][1,:]

            axarr[np.mod(j-n_dims2,3)].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
            axarr[np.mod(j-n_dims2,3)].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison') 
    
    # Ranges and labels
    # mp.fix_ticks()
    axarr[0].set_ylabel("Relative Likelihood")  
    [axarr[i].set_xlabel(nicenames[i+4+(n_mol-1)]) for i in [0,1,2]] 
    axarr[1].set_xlim([xr[0][0]+xr[1][0],xr[0][1]+xr[1][1]]) # x-axis ranges.
    axarr[2].set_xlim([xr[2][0]+xr[3][0],xr[2][1]+xr[3][1]])
    
    # Add a secondary x-axis for BACD --> Mass
    # log(mass) = log(BACD) + log(area) + log(mu)+log(m_H2) - log(X)
    #     meas['areacm'] + meas['head']['mol_weight'] + log10(2.0*units.M_p.to('kg')/units.solMass.to('kg')) - meas['head']['abundance']
    # Makes sure aligned
    ax2a.set_xlim(axarr[0].set_xlim()-np.log10(units.solLum.to('erg/s')))
    ax2a.set_xlabel(r'Log Luminosity  [L$_{\odot}$]')
    ax2b.set_xlim(axarr[2].set_xlim()+meas['addmass'])
    ax2b.set_xlabel(r'Log Molecular Mass [M$_{\odot}$]')

    plt.draw() 
    plt.savefig('fig_marginalized2.png')
    print 'Saved fig_marginalized2.png'
    
    # Plot the marginalization of the secondary parameter ratios, if 2 components.
    if n_comp==2: # JRK 1/21/16, instead of n_dims > 7, JRK 1/23/14, instead of n_dims == 8
        #mp=multipanel(dims=(1,3),figID=3,padding=(0,0),panel_size=(3,1)) #
        #if not simplify: mp.title(title,xy=(0.5,0.97))
        f, axarr=plt.subplots(ny,nx,num=3,sharey=True,figsize=(4*nx,4*ny))
        f.subplots_adjust(bottom=0.12,left=0.06,right=0.98,top=0.88)
        f.subplots_adjust(wspace=0.1)
        if not simplify: axarr[1].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
        for j in plotinds[2]:
            for m,mode in enumerate(s['modes']):
                yplot=dists[m][j][1,:]
                #mp.grid[np.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =np.ravel(zip(yplot, yplot))
                axarr[np.mod(j-n_dims,3)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j],color=modecolors[m],label='Mode')
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]+mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]-mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
            
            yplot=dists['all'][j][1,:]
            axarr[np.mod(j-n_dims,3)].plot(dists['all'][j][0,:]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
            axarr[np.mod(j-n_dims,3)].axvline(x=cube[j]+add[j],color=colors[j])
       
        if dists2:
            for j in plotinds2[2]:
                yplot2=dists2['all'][j][1,:]
                axarr[np.mod(j-n_dims2,3)].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                axarr[np.mod(j-n_dims2,3)].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')     

        axarr[0].set_ylabel("Relative Likelihood")       
        [axarr[i-min([7,8,9]+np.mod(n_dims-(n_mol-1),4))].set_xlabel(nicenames[i]) for i in [7,8,9]+np.mod(n_dims-(n_mol-1),4)]

        plt.savefig('fig_marginalized2ratio.png')
        print 'Saved fig_marginalized2ratio.png'  
    
def plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,
                 modecolors=['g','m','y','c'],simplify=False,title=''):
    import matplotlib.cm as cm
    #mp = multipanel(dims=(n_dims-1,n_dims-1),figID=4,diagonal='lower',panel_size=(3,3)) #
     
    nx=n_dims-1
    ny=n_dims-1
    f, axarr=plt.subplots(ny,nx,num=4,sharey=False,sharex=False,figsize=(3*nx,3*ny))
    f.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    f.subplots_adjust(wspace=0,hspace=0)
    if not simplify: axarr[0][0].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
    # JRK 1/21/16: Add X_mol if we have multiple molecules.
    c_plotinds=plotinds[0]
    if n_mol>1: c_plotinds=[item for sublist in [plotinds[0],plotinds[4]] for item in sublist]
    
    for g,i in enumerate(c_plotinds):
        for j in range(i):
            gridinds=(g-1,j)
            #ind=(n_dims-1) * (i-1) + j 
            #ind=(n_dims-1)-i+j+(n_dims-2)*(n_dims-1-i) # Wow.
            #mp.grid[ind].contourf
            axarr[gridinds].contourf(dists['all'][i,j][:,:,1]*mult[j]+add[j], dists['all'][i,j][:,:,0]*mult[i]+add[i], dists['all'][i,j][:,:,2], 
                    5, cmap=cm.gray_r, alpha = 0.8,origin='lower') # NEed to transpose?????
                    
            #mp.grid[ind].scatter(datsep['all'][:,j+2]*mult[j]+add[j],datsep['all'][:,i+2]*mult[i]+add[i], color='k',marker='+',s=1, alpha=0.4)
                
            if i==n_dims-1:
                axarr[gridinds].set_xlabel(parameters[j])   
            else:
                [label.set_visible(False) for label in axarr[gridinds].get_xticklabels()] 
            if j==0:
                axarr[gridinds].set_ylabel(parameters[i])
            else:
                [label.set_visible(False) for label in axarr[gridinds].get_yticklabels()]
            
            for m,mode in enumerate(s['modes']):
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
                axarr[gridinds].axhline(y=mode['mean'][i]*mult[i]+add[i],color=modecolors[m])
            
            axarr[gridinds].axvline(x=cube[j]*mult[j]+add[j],color='k',linestyle='--',label='4D Max')
            axarr[gridinds].axhline(y=cube[i]*mult[i]+add[i],color='k',linestyle='--')
            
            # If we have temperature vs. density, we have diagonal pressure lines we can add
            # Add labels indicating their values to the top of the plots
            if (parameters[j]=='h2den1' and parameters[i]=='tkin1') or  (parameters[j]=='h2den2' and parameters[i]=='tkin2'):
                presscontour=[3,4,5,6,7,8]
                for p in presscontour: 
                    axarr[gridinds].plot([axarr[gridinds].set_xlim()[0],p-axarr[gridinds].set_ylim()[0]],
                        [p-axarr[gridinds].set_xlim()[0],axarr[gridinds].set_ylim()[0]],':k')
                    axarr[gridinds].annotate('{:.0f}'.format(p),xy=(p-axarr[gridinds].set_ylim()[1],axarr[gridinds].set_ylim()[1]),
                        xycoords='data')
            # Likewise for column density and filling factor
            if (parameters[j]=='cdmol1' and parameters[i]=='ff1') or  (parameters[j]=='cdmol2' and parameters[i]=='ff2'):
                bacdcontour=[15,16,17,18,19,20,21,22,23]
                for p in bacdcontour: 
                    axarr[gridinds].plot([axarr[gridinds].set_xlim()[0],p-axarr[gridinds].set_ylim()[0]],
                        [p-axarr[gridinds].set_xlim()[0],axarr[gridinds].set_ylim()[0]],':k')
                    axarr[gridinds].annotate('{:.0f}'.format(p),xy=(p-axarr[gridinds].set_ylim()[1],axarr[gridinds].set_ylim()[1]),
                        xycoords='data')
        
        # Remove the unused boxes in this row, which is j=i through j=n_dims-2
        if g>0:
            for j in range(i,n_dims-1): axarr[(g-1,j)].axis('off')

    ###### Still need to get everything aligned right.  Fix ticks moves the outer boxes
    #mp.fix_ticks()
    for g,i in enumerate(c_plotinds):
        for j in range(i):
            #ind=(n_dims-1)-i+j+(n_dims-2)*(n_dims-1-i) # Wow.
            gridinds=(g-1,j)
            #print np.min(dists['all'][i,j][:,:,1]), np.max(dists['all'][i,j][:,:,1]),mult[j],add[j]
            #print np.min(dists['all'][i,j][:,:,0]), np.max(dists['all'][i,j][:,:,0]),mult[j],add[j]
            axarr[gridinds].set_xlim([np.min(dists['all'][i,j][:,:,1]*mult[j]+add[j]),np.max(dists['all'][i,j][:,:,1]*mult[j]+add[j])])
            axarr[gridinds].set_ylim([np.min(dists['all'][i,j][:,:,0]*mult[i]+add[i]),np.max(dists['all'][i,j][:,:,0]*mult[i]+add[i])])
    
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()

    plt.savefig('fig_conditional.png')
    print 'Saved fig_conditional.png'

def plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,
                 modecolors=['g','m','y','c'],title='',simplify=False):
    import matplotlib.cm as cm
    #nplot=n_sec[0]-1
    #mp = multipanel(dims=(nplot,nplot),figID=5,diagonal=True,panel_size=(3,3)) # 
    
    nx=n_sec[0]-1
    ny=nx
    f, axarr=plt.subplots(ny,nx,num=5,sharey=False,sharex=False,figsize=(2*nx,2*ny))
    f.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    f.subplots_adjust(wspace=0,hspace=0)
    if not simplify: axarr[0][0].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
    for g,i in enumerate(plotinds[1]):
        for h,j in enumerate(range(n_dims,i)):
            gridinds=(g-1,h)
            #ind=(nplot) * (i-n_dims-1) + j-n_dims
            #ind=(nplot)-(i-n_dims)+(j-n_dims)+(nplot-1)*(nplot-(i-n_dims))
            #mp.grid[ind]
            axarr[gridinds].contourf(dists['all'][i,j][:,:,1]+add[j], dists['all'][i,j][:,:,0]+add[i], dists['all'][i,j][:,:,2], 
                    5, cmap=cm.gray_r, alpha = 0.8,origin='lower') # NEed to transpose?????
                
            if g==nx:
                axarr[gridinds].set_xlabel(parameters[j])
            else:
                [label.set_visible(False) for label in axarr[gridinds].get_xticklabels()] 
            if h==0:
                axarr[gridinds].set_ylabel(parameters[i])
            else:
                [label.set_visible(False) for label in axarr[gridinds].get_yticklabels()]
 
            for m,mode in enumerate(s['modes']):
                axarr[gridinds].axvline(x=mode['mean'][j]+add[j],color=modecolors[m],label='Mode')
                axarr[gridinds].axhline(y=mode['mean'][i]+add[i],color=modecolors[m])
            
            axarr[gridinds].axvline(x=cube[j]+add[j],color='k',linestyle='--',label='4D Max')
            axarr[gridinds].axhline(y=cube[i]+add[i],color='k',linestyle='--')
        
        # Remove blank boxes in upper diagonal
        if g>0: 
            for h in range(g,nx): axarr[(g-1,h)].axis('off')
            
           
    ###### Still need to get everything aligned right.
    #mp.fix_ticks()
    for g,i in enumerate(plotinds[1]):
        for h,j in enumerate(range(n_dims,i)):
            gridinds=(g-1,h)
            #ind=(nplot)-(i-n_dims)+(j-n_dims)+(nplot-1)*(nplot-(i-n_dims))
            axarr[gridinds].set_xlim([np.min(dists['all'][i,j][:,:,1]+add[j]),np.max(dists['all'][i,j][:,:,1]+add[j])])
            axarr[gridinds].set_ylim([np.min(dists['all'][i,j][:,:,0]+add[i]),np.max(dists['all'][i,j][:,:,0]+add[i])])

            # Remove if either one of these has luminosity not calculated.
            if np.max(dists['all'][j][1,:])==np.min(dists['all'][j][1,:]):
                axarr[gridinds].annotate('Luminosity \nNot Calculated',xy=(0.1,0.5),xycoords='axes fraction')
                [label.set_visible(False) for label in axarr[gridinds].get_xticklabels()] 
            if np.max(dists['all'][i][1,:])==np.min(dists['all'][i][1,:]):
                # axarr[gridinds].axis('off') Don't do this, we might need the axis labels of the other parameter
                [label.set_visible(False) for label in axarr[gridinds].get_yticklabels()] 
                
            
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()
        

    plt.savefig('fig_conditional2.png')
    print 'Saved fig_conditional2.png'
    
def plotsled(meas,cube,n_params,n_dims,n_comp,modemed,modemax,modesig,plotinds,title='',lum=False,meas2=0,cube2=[],setyr=[0,0],
             colors=['b','r'],colors2=['#6495ED','m'],n_dims2=0,n_comp2=0,simplify=False,asymerror=0,
             sled_to_j=0,newplot=True,legend=True,mol='',alpha=0.2,plotmax=True,linestyle='-'):
    # 12/3/13: Fixed bug if meas2 and meas have different number of elements (esp. diff # of non-upper-limits).
    # 4/2/14: Use asymerror=Table if you want to use the assymetric error regions for plotting.
    # 8/27/15: Set newplot=False to plot to an existing figure, and not save the png. Also skips the next plots.
    #    Can also set legend=False if want to exclude the legend (for same purpose, for example)
    #    y axis limit is now set using only those points of values > 0 and S/N > 3 (else too small ylimit)
    # 1/21/16: Use n_comp==2 instead of n_dims==8. Get rid of the n_dims mod 4 stuff, which was related
    #   to the dust correction, which is not used here.
    # 1/22/16: Allow input mol=string, to do secondary molecules. If blank, do primary molecule.
    # 3/26/16: Allow input alpha, the alpha value for the SLED likelihoods. Default 0.2.
    #    Allow plotmax=False to not plot the maximum likelihood lines (which calls RADEX). Default True.
    
    if mol=='': mol=meas['head']['mol']
    
    if newplot:
        plt.figure(6,figsize=(10,8))
        plt.subplots_adjust(bottom=0.08,left=0.08,right=0.97,top=0.97) # LEFT 0.12 JRK 4/6/14
        plt.clf()
    plt.xlabel('Upper J')
    if lum:
        plt.ylabel(r'Luminosity [L$_\odot$]')
    else: 
        plt.ylabel('K (per km/s)')
    plt.yscale('log')
    if not simplify: plt.title(title)
   
    # First, plot the data. 1/21/16: for given molecule only.
    # 3/25/16: For backwards compatibility, if there is no meas['mol'], add it.
    if 'mol' not in meas.keys():
        meas['mol']=np.array([meas['head']['mol'] for x in meas['flux']])
    ok=np.all([meas['flux']!=0,meas['mol']==mol],axis=0)#[meas['flux']!=0]
    ul=np.all([meas['flux']==0,meas['mol']==mol],axis=0)#[meas['flux']==0]

    xplot=meas['J_up']
    yplot=meas['flux']
    yerr=meas['sigma']
    yplot0=yplot
    yerr0=yerr
    
    # Did we do optical depth corrections?  If so, plot: the data with the MEAN lambda0 and beta results AND data with the input 
    #  lambda0 and beta, AND of course the original data.
    #if np.mod(n_dims,4)==2:
    #    label=['Data (Median Corr)','Data (Input Corr)','Data (No Corr)']
    #    xplot=[xplot,xplot,xplot]
    #    corr1=tau_corr(meas['J_up'],modemed[n_dims-2],modemed[n_dims-1])
    #    corr2=tau_corr(meas['J_up'],meas['head']['beta'],meas['head']['lambda0'])
    #    yplot=[yplot/corr1,yplot/corr2,yplot]
    #    yerr=[yerr/corr1,yerr/corr2,yerr]
    #else: 
    label=['Data']
    xplot=[xplot]
    yplot=[yplot]
    yerr=[yerr]
    
    # And if we have the SLED likelihoods... those are in plotinds[3].  ONLY for the run we are plotting... no comparison.
    if mol==meas['head']['mol']: 
        if sled_to_j: sledinds=plotinds[3] # if n_dims==8 else plotinds[2], changed 1/21/16
        xmolind1=0
        xmolind2=0
    else:
        if sled_to_j: sledinds=plotinds[5+np.where(np.array(meas['head']['secmol'])==mol)[0][0]]
        xmolind1=n_comp*4+np.where(np.array(meas['head']['secmol'])==mol)[0][0]
        xmolind2=n_comp*4+(len(meas['head']['secmol']))+np.where(np.array(meas['head']['secmol'])==mol)[0][0]

    if sled_to_j:
        if asymerror:
            yfill1=asymerror['-1 Sigma'][sledinds]
            yfill2=asymerror['+1 Sigma'][sledinds]
        else: 
            yfill1=modemed[sledinds]-modesig[sledinds]
            yfill2=modemed[sledinds]+modesig[sledinds]            
        
        # If the lower one is zero or below, the shape will come out completely wrong on this log plot.
        yfill1[yfill1 <= 0]=1e-10
        yfill2[yfill2 <= 0]=2e-10
        
        if lum:     
            from pysurvey import spire_conversions
            if 'dl' not in meas['head']: 
                print "YOU MUST RERUN MEASDATA_PICKLE SO THAT MEAS['HEAD']['DL'] IS INCLUDED"
                print "Probably crashing now..."
            
            (yfill1,trash)=spire_conversions(yfill1,'kkms','lsol',range(1,14,1)*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=meas['head']['dl'])
            (yfill2,trash)=spire_conversions(yfill2,'kkms','lsol',range(1,14,1)*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=meas['head']['dl'])       
            yfill1*=meas['head']['lw']
            yfill2*=meas['head']['lw']
        
        if n_comp==1:    
            plt.fill_between(range(1,sled_to_j+1,1), yfill1, yfill2, where=yfill2>=yfill1, 
                 facecolor='gray', interpolate=True,alpha=alpha)
        else:
            plt.fill_between(range(1,sled_to_j+1,1), yfill1[0:sled_to_j], yfill2[0:sled_to_j], 
                where=yfill2[0:sled_to_j]>=yfill1[0:sled_to_j], facecolor='b', interpolate=True,alpha=alpha)
            plt.fill_between(range(1,sled_to_j+1,1), yfill1[sled_to_j:], yfill2[sled_to_j:], 
                where=yfill2[sled_to_j:]>=yfill1[sled_to_j:], facecolor='r', interpolate=True,alpha=alpha)

    if lum:
        from pysurvey import spire_conversions
        import astropy.coordinates.distances as dist
        (yplot,yerr)=spire_conversions(yplot,'kkms','lsol',xplot*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=meas['head']['dl'],inerr=yerr)
        yplot*=meas['head']['lw']
        yerr*=meas['head']['lw']              
    
    for j in range(len(xplot)):
        plt.errorbar(xplot[j][ok],yplot[j][ok],yerr=yerr[j][ok],
                 color=['black','green','purple'][j],fmt='^',linestyle='None',label=label[j])
    
        arrlen =arrowloglen(plt.ylim(),3.0*yerr[j][ul]       ,frac=32.0)
        headlen=arrowloglen(plt.ylim(),3.0*yerr[j][ul]+arrlen,frac=64.0)
    
        for i, l in enumerate(arrlen):
            plt.arrow(xplot[j][ul][i], 3.0*yerr[j][ul][i], 0.0*xplot[j][ul][i], l, 
                  fc=['black','green','purple'][j], ec=['black','green','purple'][j], head_width=0.3, head_length=-headlen[i],overhang=1) 
                  
    if meas2: 
        ok2=np.all([meas2['flux']!=0,meas2['mol']==mol],axis=0)#[meas2['flux']!=0]
        ul2=np.all([meas2['flux']==0,meas2['mol']==mol],axis=0)#[meas2['flux']==0]

        xplot2=meas2['J_up']
        yplot2=meas2['flux']
        yerr2=meas2['sigma']
        if lum:
            (yplot2,yerr2)=spire_conversions(yplot2,'kkms','lsol',xplot2*115.3,sr=meas2['head']['omega_s'],
                              z=meas2['head']['z'],dist=meas2['head']['dl'],inerr=yerr2)
            yplot2*=meas2['head']['lw']
            yerr2*=meas2['head']['lw']
    
        plt.errorbar(xplot2[ok2],yplot2[ok2],yerr=yerr2[ok2],
                 color='gray',fmt='o',linestyle='None',label='Comparison Data')
    
        arrlen2 =arrowloglen(plt.ylim(),3.0*yerr2[ul2]       ,frac=32.0)
        headlen2=arrowloglen(plt.ylim(),3.0*yerr2[ul2]+arrlen2,frac=64.0)
    
        for i, l in enumerate(arrlen2):
            plt.arrow(xplot2[ul2][i], 3.0*yerr2[ul2][i], 0.0*xplot2[ul2][i], l, 
                      fc='gray', ec='gray', head_width=0.3, head_length=-headlen2[i],overhang=1) 
    
    # Next, plot 4DMax, then the mode, then Mode Maximum.
    # If doing a comparison, only do 4DMax...
    w_in=[0,1,2]
    if meas2: w_in=[0,3]
    if simplify and not meas2: w_in=[0]
    if plotmax==False: w_in=[]
    chisq=[]
    t_n_params=n_params # Temporary n_params, incase we have to replace it with 2
    t_n_dims=n_dims
    t_n_comp=n_comp
    t_colors=colors
    for w in w_in:
        if w==0: 
            thiscube=cube
            thismeas=meas
            thisok=ok
            linestyle=linestyle
            label1='Component 1'  # JRK 4/6/14 'Low-Pressure', 'High-Pressure'
            label2='Component 2'
            label3='Component 3'
            labelT='Total (Best Fit)'
        if w==1: 
            thiscube=modemax
            thismeas=meas
            thisok=ok
            linestyle='--'
            label1=''
            label2=''
            label3=''
            labelT='Total (Mode Maximum)'
        if w==2: 
            thiscube=modemed
            thismeas=meas
            thisok=ok
            linestyle=':'
            label1=''
            label2=''
            label3=''
            labelT='Total (Mode Median)'
        if w==3:
            thiscube=cube2  
            thismeas=meas2
            thisok=ok2
            linestyle='--'
            label1=''
            label2=''
            lable3=''
            labelT='Comparison Total (Best Fit)'
            t_n_params=len(cube2)
            t_n_dims=n_dims2
            t_colors=colors2
            t_n_comp=n_comp2
        
        logcol1=cube[2]
        if xmolind1!=0: logcol1+=cube[xmolind1]
        dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,logcol1), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=meas['tbg'], species=mol, velocity_gradient=1.0, 
                          debug=False, return_dict=True)
        dat['FLUX_Kkms']=np.array(map(float,dat['FLUX_Kkms']))*np.power(10,thiscube[3])
        model1=dat['FLUX_Kkms']
        
        #if np.mod(n_dims,4)==2: model1/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
        if lum: 
            (model1,trash)=spire_conversions(model1,'kkms','lsol',dat['J_up']*115.3,sr=thismeas['head']['omega_s'],
                              z=thismeas['head']['z'],dist=dist.Distance(z=thismeas['head']['z']).Mpc)
            model1*=thismeas['head']['lw']
        plt.plot(dat['J_up'],model1,color=t_colors[0],label=label1,linestyle=linestyle,marker=None)
        newdat=dat['FLUX_Kkms']
    
        if t_n_comp==2:
            logcol2=cube[6]
            if xmolind2!=0: logcol2+=cube[xmolind2]
            dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
                           temperature=np.power(10,thiscube[5]), column=np.power(10,logcol2), 
                           collider_densities={'H2':np.power(10,thiscube[4])},
                           tbg=meas['tbg'], species=mol, velocity_gradient=1.0, 
                           debug=False,return_dict=True)
            dat2['FLUX_Kkms']=np.array(map(float,dat2['FLUX_Kkms']))*np.power(10,thiscube[7])
            model2=dat2['FLUX_Kkms']
            #if np.mod(n_dims,4)==2: model2/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
            if lum:
                (model2,trash)=spire_conversions(model2,'kkms','lsol',dat2['J_up']*115.3,sr=thismeas['head']['omega_s'],
                              z=thismeas['head']['z'],dist=dist.Distance(z=thismeas['head']['z']).Mpc)
                model2*=thismeas['head']['lw']  

            model3=model1+model2
            newdat=np.array(map(float,dat['FLUX_Kkms']))+np.array(map(float,dat2['FLUX_Kkms']))
            plt.plot(dat2['J_up'],model2,color=t_colors[1],label=label2,linestyle=linestyle,marker=None)
            plt.plot(dat2['J_up'],model3,color='k',label=labelT,linestyle=linestyle,marker=None)           


        # Calculate Chi Squared?
        # Need to match "newdat" with meas['flux']
        #if np.mod(n_dims,4)==2: newdat/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
        chisq.append(0)
        for i,tflux in enumerate(thismeas['flux'][thisok]):
            temp=newdat[dat['J_up'] == thismeas['J_up'][thisok][i]]
            #print thismeas['J_up'][thisok][i],tflux,thismeas['sigma'][thisok][i],temp
            if w==3:
                chisq[1]+=((temp-tflux)/thismeas['sigma'][thisok][i])**2.0
                #print temp
            else:
                chisq[w]+=((temp-tflux)/thismeas['sigma'][thisok][i])**2.0
                #print temp
      
    # A bit more tweaking of the plot:
    plt.xlim([0,14])
    #plt.ylim([1e-3,1e1])
    sigmaylim=3.0
    ylimuse=np.all([yplot0>0,yplot0/yerr0>3.0],axis=0)
    while np.sum(ylimuse)==0:  # In case we don't have any real 3 sigma...
        sigmaylim-=0.5
        ylimuse=np.all([yplot0>0,yplot0/yerr0>sigmaylim],axis=0)
    
    if meas2:
        plt.ylim([np.min(np.concatenate((yplot0[yplot0>0]-1.0*yerr0[yplot0>0],yplot2[yplot2>0]-1.0*yerr2[yplot2>0]),axis=0)),
              np.max(np.concatenate((yplot0[yplot0>0]+1.0*yerr0[yplot0>0],yplot2[yplot2>0]+1.0*yerr2[yplot2>0]),axis=0))])
    else: 
       plt.ylim([np.min(yplot0[ylimuse]-3.0*yerr0[ylimuse]),
              np.max(yplot0[ylimuse]+3.0*yerr0[ylimuse])])
    # Overrride that ylim if setyr is used.
    if np.sum(setyr) != 0: plt.ylim(setyr)
    yr=np.log10(plt.ylim())
    if not simplify: # only print Chi Sq if simplify is not True.
        for i,c in enumerate(chisq): 
            pos=np.power(10,(0.2-0.05*i)*(yr[1]-yr[0])+yr[0])
            plt.text(2,pos,c)
    if legend: plt.legend()
    
    if newplot: 
        plt.savefig('fig_sled_'+mol+'.png')
        print 'Saved fig_sled_'+mol+'.png'
    else:
        return chisq
    
    ######## OPTICAL DEPTH
    plt.figure(7)
    plt.clf()
    plt.xlabel('Upper J')
    plt.ylabel('Tau')
    if not simplify: plt.title(title)

    # Ugh, dat is whatever was last used, must call RADEX again...
    thiscube=cube
    thismeas=meas
    logcol1=cube[2]
    if xmolind1!=0: logcol1+=cube[xmolind1] 
       
    dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,logcol1), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=meas['tbg'], species=mol, velocity_gradient=1.0, 
                          debug=False,return_dict=True)    
    
    if n_comp==2: 
        logcol2=cube[6]
        if xmolind2!=0: logcol2+=cube[xmolind2]
        dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
                           temperature=np.power(10,thiscube[5]), column=np.power(10,logcol2), 
                           collider_densities={'H2':np.power(10,thiscube[4])},
                           tbg=meas['tbg'], species=mol, velocity_gradient=1.0, 
                           debug=False,return_dict=True)
    
    plt.plot(dat['J_up'],dat['TAU'],color=colors[0],label='Component 1')
    if n_comp==2: plt.plot(dat2['J_up'],dat2['TAU'],color=colors[1],label='Component 2')

    plt.xlim([0,14])
    plt.legend()

    plt.savefig('fig_tau_'+mol+'.png')
    print 'Saved fig_tau_'+mol+'.png'
    
    ######## EXCITATION TEMPERATURE
    plt.figure(8)
    plt.clf()
    plt.xlabel('Upper J')
    plt.ylabel('Excitation Temperature [K]')
    if not simplify: plt.title(title)

    plt.plot(dat['J_up'],dat['T_EX'],color=colors[0],label='Component 1')
    plt.axhline(y=np.power(10,cube[1]),color=colors[0],linestyle='--')
    if n_comp==8: 
        plt.plot(dat2['J_up'],dat2['T_EX'],color=colors[1],label='Component 2')
        plt.axhline(y=np.power(10,cube[5]),color=colors[1],linestyle='--')

    plt.xlim([0,14])
    #plt.ylim([0,1e3])
    plt.legend()

    plt.savefig('fig_tex_'+mol+'.png')
    print 'Saved fig_tex_'+mol+'.png'
    
def plotmarginalsled(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,n_comp,nicenames,n_mol,title='',norm1=True,
                 modecolors=[[0,0],[0.1,0.9],[0.2,0.5],[0.3,0.7,1]],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=0,simplify=False,useind=3,mol=''):

    # JRK 1/22/16: Two new inputs, useind=3 and mol='' is default behavior for plotting the 
    #  primary molecule. Changing useind to 5, 6, etc. will plot the likelihoods for secondary molecules.
    #  Putting a string for mol will add the molecule name to the filename.
    
    sled_to_j=len(plotinds[useind]) # Was useind=3 if n_dims>7, 2 if not, but now just 3. 1/21/16
    if n_comp==2: sled_to_j=sled_to_j/2
    nx=int(np.ceil(np.sqrt(sled_to_j)))
    ny=int(np.ceil(sled_to_j/np.ceil(np.sqrt(sled_to_j))))
    
    plt.figure(9,figsize=(4*nx,4*ny))
    plt.clf()
    f, axarr=plt.subplots(ny,nx,num=9, sharey=True,figsize=(4*nx,4*ny))
    f.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    f.subplots_adjust(wspace=0.1)
    if not simplify: axarr[0][0].annotate(title,xy=(0.535,0.97),horizontalalignment='center',xycoords='figure fraction')
    
    for g,j in enumerate(plotinds[useind]):      
        gridinds=np.floor((np.mod(g,sled_to_j))/nx),np.mod(np.mod(g,sled_to_j),nx)
        if not dists2: # Don't do these if we are overplotting a 2nd dist, too confusing.
            for m,mode in enumerate(s['modes']):
                if colors[j]=='b': 
                    modecol=[modecolors[m][0],modecolors[m][1],1]
                elif colors[j]=='r':
                    modecol=[1,modecolors[m][1],modecolors[m][0]]
                elif colors[j]=='g': # Added JRK 1/23/14
                    modecol=[modecolors[m][0],1,modecolors[m][1]]
                elif colors[j]=='k':
                    modecol=[modecolors[m][0],modecolors[m][0],modecolors[m][0]]
                
                yplot=dists[m][j][1,:]
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =np.ravel(zip(yplot, yplot))
                axarr[gridinds].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m

                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                axarr[gridinds].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                ##print mode['maximum'][j]+add[j],mode['sigma'][j]
           
        yplot=dists['all'][j][1,:] 
        axarr[gridinds].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        axarr[gridinds].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
        
        if norm1: axarr[gridinds].set_ylim(0,1)
        if g<=sled_to_j-1: 
            axarr[gridinds].set_xlabel(nicenames[np.mod(g,sled_to_j)+4+(n_mol-1)+6]) # Fixed 6/2/16; 6 no matter what, not dep on ncomp
        if gridinds[1]==0: axarr[gridinds].set_ylabel("Relative Likelihood")
        
    if dists2:
        sled_to_j2=len(plotinds2[3])
        if n_dims >7: sled_to_j2=sled_to_j2/2
        try: 
            for g,j in enumerate(plotinds2[3]):
                gridinds=np.floor((np.mod(g,sled_to_j2))/nx),np.mod(g,nx)
                yplot2=dists2['all'][j][1,:]
                axarr[gridinds].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                axarr[gridinds].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')          
        except:
            print 'No SLED distribution comparison to overplot'  
    
    # Remove unused axes.
    if sled_to_j<nx*ny:
        for i in -1*(np.arange(nx*ny-sled_to_j)+1): axarr[ny-1][i].axis('off')
    
    pltfile='fig_marginalizedsled.png' if mol=='' else 'fig_marginalizedsled_'+mol+'.png'
    plt.savefig(pltfile)
    print 'Saved '+pltfile

#####################################################################################
def get_covariance(datsep):
    # To find the correlation coefficients between parameters, and 
    # get the full covariance matrix.
    #
    # C = ( sigma_x^2                sigma_x*sigma_y*rho_xy 
    #       sigma_x*sigma_y*rho_xy     sigma_y^2 )
    # R  =  (  1     rho_xy
    #       rho_xy    1   )
    #     
    # Though this does over ALL parameters, keep in mind that some are LOG of a parameter.
    # 
    # Mode Mean: Weighted average of points in mode = Sum_i (x_i * weight_i) for all points in mode.
    # Mode Sigma: Sqrt(weighted sample variance) = SQRT (  Sum_i (x_i^2 * weight_i) - Mean^2)
    # Use rho_xy = ( <xy> - <x><y> ) / SQRT( (<x^2> - <x>^2)(<y^2> - <y>^2) )
    # Columns of the data have sample probability (weight), -2*loglikehood, n samples
    # Do for each mode, and total marginalized, just like we do the distributions.
    print 'Finding covariance for modes: ',datsep.keys()
    cov={}
    n=len(datsep['all'][0])-2
    for key in datsep.keys():
        r_arr=np.empty([n,n])
        c_arr=np.empty([n,n])
        for i in range(n):
            x=np.sum(datsep[key][:,i+2]*datsep[key][:,0])
            x2=np.sum(np.power(datsep[key][:,i+2],2)*datsep[key][:,0])
            sx=np.sqrt(x2 - x**2)
            for j in range(n): # Yes, it is redundant because array is symmetric.
                y=np.sum(datsep[key][:,j+2]*datsep[key][:,0])
                y2=np.sum(np.power(datsep[key][:,j+2],2)*datsep[key][:,0])
                sy=np.sqrt(y2-y**2)
                xy=np.sum(datsep[key][:,j+2]*datsep[key][:,i+2]*datsep[key][:,0])
                r=(xy-x*y)
                r/=np.sqrt( (x2-x**2)*(y2-y**2))
                r_arr[i,j]=r
                c_arr[i,j]=r*sx*sy
        cov[key]={}
        cov[key]['r']=r_arr
        cov[key]['c']=c_arr
    
    return cov
    
def fix_flux_modemean(s,datsep,plotinds,useind=3):
    # Rarely, some versions of RADEX might produce a non-sensical flux for a line, e.g.
    # of the order 10^50. If such a value is included in a "mode" in parameter space,
    # the "modemean" and "modesigma" will be significantly skewed by this value, despite
    # the low probability of that point. This does not seem to affect the median values, 
    # which are derived from the CDF.
    # modemean(x) = Sum_i (x_i * weight_i) for all points in mode.
    # modesigma(x) = SQRT (  Sum_i (x_i^2 * weight_i) - Mean^2)
    # Added useind. Will be 3 for main molecule, but higher 5+ for secondary molecules
    nmodes=len(s['modes'])
    for m in range(nmodes):
        fluxes=[s['modes'][m]['mean'][x] for x in plotinds[useind]] # 1/21/16, was 2, should be 3
        for i, f in enumerate(fluxes):
            # Redo calculations for all fluxes.
            x=datsep[m][:,plotinds[useind][i]+2]
            w=datsep[m][:,0]
            good=[np.abs(x-np.median(x))<1e10]
            print 'Replaced modemean modesigma ',s['modes'][m]['mean'][plotinds[useind][i]], s['modes'][m]['sigma'][plotinds[useind][i]]
            mean=np.sum(x[good]*w[good])
            sigma=np.sqrt(np.sum(x[good]**2*w[good])-mean**2)
            s['modes'][m]['mean'][plotinds[useind][i]]=mean
            s['modes'][m]['sigma'][plotinds[useind][i]]=sigma
            print 'with ',mean,sigma
            
        
def nbyn(nvars):
    # nrow,ncol,unused=nbyn(nvars)
    # For a given number of parameters (nvars) that you want to plot, determine the 
    # ideal array configuration of plots and the indices of the ones unused.
    # Get rid of those using: 
    # for i in unused:
    #    ind=np.unravel_index(i,axarr.shape)
    #    axarr[ind].axis('off')
    # Create your axis array using:
    #  fig,axarr=plt.subplots(nrow,ncol,num=fignum)
    # and then for a given parameter, the axis index is:
    #  ind=np.unravel_index(i,axarr.shape) # Use axarr[ind]
    
    nrow=int(np.ceil(np.sqrt(nvars)))
    ncol=nrow if nrow>np.mod(nrow**2,nvars) else nrow-1
    if nvars==2: ncol=1

    return nrow,ncol,range(nvars,ncol*nrow)