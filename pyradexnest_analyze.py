#
import pymultinest
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import cPickle as pickle
from astropy.table import Table, Column
import sys
from pyradexnest_analyze_tools import *
from config import *

# 1/14/16: Copied from pyradexnest_analyze_multimol.py
#    Meaning to make it compatible with normal single molecule version.
# 5/26/16: Added some keywords to plotting routines. Correctly calculate xranges
#    for secondary species xmol parameters.

#################################################################################

simplify=True

#### LOAD THE STATS
title=os.getcwd()
title=title.split('/')[-1]

### Backwards compatibility. Previously, config file just had n_dims.
# Now, config file has n_comp and n_mol, from which n_dims is calculated (also included)
try:
    n_mol
except:
    n_mol=1
    n_comp=1 if n_dims==4 else 2
    
# Total number of parameters: Dimensions, "Secondary Parameters", SLED Likelihoods
if n_comp==2:
    n_sec=[6,3]
    n_sled=2*sled_to_j*n_mol
else:
    n_sec=[3]
    n_sled=sled_to_j*n_mol
n_params =n_dims + np.sum(n_sec) + n_sled

meas=pickle.load(open("measdata.pkl","rb"))
lw=np.log10(meas['head']['lw'])
# If meas doesn't include tbg, just the old default, 2.73 K
if 'tbg' not in meas: meas['tbg']=2.73
# If not calculated using multimol, won't have secmol.
if 'secmol' not in meas['head']: meas['head']['secmol']=[]

a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()
data= a.get_data()
# Check if a.post_file exists; this separates the data by mode.
#### TEMPORARY FIX, in case old version of pymultinest with typo is being used.
if a.post_file==u'chains/1-post_seperate.dat': a.post_file=u'chains/1-post_separate.dat'
if os.path.isfile(a.post_file):
    datsep=post_sep(a.post_file)  # Divide the "data" up by mode.
else:
    datsep={}
    datsep[0]=data
datsep['all']=data
bestfit=a.get_best_fit()
cube=bestfit['parameters'] # The best fit is not the same as the mode, cube=s['modes'][0]['maximum']
nmodes=len(s['modes'])

#### PLOT SETTINGS
# Get the correct plot colors, factors to add, indices to use, etc.
[parameters,add,mult,colors,plotinds,sledcolors]=define_plotting_multimol(n_comp,n_mol,n_dims,n_sec,n_params,sled_to_j,lw)
modecolors=['g','m','y','c','k','r','b']
for x in plotinds: print [parameters[y] for y in x] # A quick check of plot indices, 6/2/2016
    
nicenames=[r'log(n$_{H2}$ [cm$^{-3}$])',r'log(T$_{kin}$ [K])',r'log(N$_{CO}$ [cm$^{-2}$])',r'log($\Phi$)',
           r'log(L[erg s$^{-1}$])',r'log(Pressure [K cm$^{-2}$])',r'log(<N$_{CO}$> [cm$^{-2}$]',
           r'log(Ratio L$_{warm}$/L$_{cold}$)',r'log(Ratio P$_{warm}$/P$_{cold}$)',r'log(Ratio <N>$_{warm}$/<N>$_{cold}$)']
if sled_to_j:
    for x in range(sled_to_j): nicenames.append(r'Flux J='+str(x+1)+'-'+str(x)+' [K]')
# Insert "nicenames" if multimol. Preserve correct order! Fixed 6/2/2016
for i,secmol in enumerate(meas['head']['secmol']):
    tmp=nicenames.insert(4+i,r'X$_{'+secmol+'/'+meas['head']['mol']+'}$')

# Check if we need to fix a completely absurd modemean in the fluxes from radex
if sled_to_j: 
    fix_flux_modemean(s,datsep,plotinds)
    # Addition, do this for the secondary molecules as well. 6/2/2016
    for i,secmol in enumerate(meas['head']['secmol']):
        print [parameters[y] for y in plotinds[5+i]]
        fix_flux_modemean(s,datsep,plotinds,useind=5+i)

# Determine plot xrange from the prior limits.
xrange=np.ndarray([n_dims,2])
xrange[:,0]=0
xrange[:,1]=1
myprior(xrange[:,0],n_dims,n_params)
myprior(xrange[:,1],n_dims,n_params)

# Squash it down if we have 2 components.
if n_comp==2:
    for i in range(4):
        xrange[i,0]=min(xrange[i,0],xrange[i+4,0])
        xrange[i,1]=max(xrange[i,1],xrange[i+4,1])
    for i in range(n_mol-1):
        xrange[i+8,0]=min(xrange[i+8,0],xrange[i+8+n_mol-1,0])
        xrange[i+8,1]=max(xrange[i+8,1],xrange[i+8+n_mol-1,1])
    # Okay I got lazy here.
    if n_mol==2: 
        xrange=xrange[[0,1,2,3,8]]
    elif n_mol==3:
        xrange=xrange[[0,1,2,3,8,10]]
    elif n_mol==4:
        xrange=xrange[[0,1,2,3,8,10,12]]

# Add linewidth to column density range
xrange[2,:]+=lw

######################################

# If a binned pickle already exists and is more 
#   recent than chains, use it.
# This is because binning takes time, and you don't want to 
# redo it if you are just replotting.
# Otherwise, do all the binning and save it to a pickle.

distfile='distributions.pkl'
dists=get_dists(distfile,s,datsep,n_dims + np.sum(n_sec),grid_points=40)

# Sanity check on the distributions.
nrow,ncol,unused=nbyn(n_params)
fig,axarr=plt.subplots(nrow,ncol,num=0,sharex=False,sharey=False,figsize=(4*ncol,4*nrow))
for x in np.arange(0,n_params):
    ind=np.unravel_index(x,axarr.shape)
    axarr[ind].plot(dists['all'][x][0],dists['all'][x][1],color=colors[x])
    axarr[ind].set_xlabel(parameters[x])
for i in unused:
    ind=np.unravel_index(i,axarr.shape)
    axarr[ind].axis('off')
fig.tight_layout()
fig.savefig('fig_raw.png')
print 'Saved fig_raw.png'
    

######################################
# Table.

table=maketables(s,n_params,parameters,cube,add,mult,n_comp,title=title,addmass=meas['addmass'],n_dims=n_dims)
modemed=table['Mode Mean'][0:n_params]-add      # Mass is last; not included.
modemax=table['Mode Maximum'][0:n_params]-add
modesig=table['Mode Sigma'][0:n_params]
pickle.dump(table, open('table.pkl', "wb") )


# Could also get covariance.  Nothin' doin' with yet.
# cov=get_covariance(datsep)

######################################
# Comparisons
# Not implemented yet.  
#  Note you'll also add in the normalization below...

######################################
# Normalization of distributions.
if norm1: 
    for mode in dists.keys():
        for p in range(n_params):
            dists[mode][p][1,:]/=dists['all']['max'][p]
            if mode != 'all': 
                # Stop changing keys on me...
                try:
                    dists[mode][p][1,:]*=np.exp(np.array(s['modes'][mode]['local evidence']))/np.exp(s['global evidence'])
                except:
                    dists[mode][p][1,:]*=np.exp(np.array(s['modes'][mode][u'local log-evidence']))/np.exp(s['global evidence'])


######################################
# Plots

doplot=True
# For now....
if doplot:
    plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,
        xr=xrange,title=title,norm1=norm1,colors=colors,simplify=simplify)    
    plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_comp,n_sec,n_dims,nicenames,n_mol,
        xr=xrange,title=title,norm1=norm1,colors=colors,meas=meas,simplify=simplify)
    
    plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,title=title,simplify=simplify)
    plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,n_mol,title=title,simplify=simplify)
    
    if n_mol>1:
        molcolors=np.array(['k' for i in range(n_comp*n_mol)])
        if n_comp==2:
            molcolors[0:n_mol]='b'
            molcolors[n_mol:]='r'
        plotmarginalxmol(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,n_mol,nicenames,
                colors,xr=xrange,title=title,simplify=simplify)
    
    plotsled(meas,cube,n_params,n_dims,n_comp,modemed,modemax,modesig,plotinds,
        title=meas['head']['mol'],sled_to_j=sled_to_j,simplify=simplify)
    for secmol in meas['head']['secmol']:
            plotsled(meas,cube,n_params,n_dims,n_comp,modemed,modemax,modesig,plotinds,
                title=secmol,sled_to_j=sled_to_j,simplify=simplify,mol=secmol)   
    
    if sled_to_j: 
        # First, the primary molecule
        plotmarginalsled(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,n_comp,nicenames,n_mol,
            title=title+' '+meas['head']['mol'],colors=colors,simplify=simplify,
            useind=3) # For now, do not add the molecule to filename, to keep consistent with non multi version.
        # All secondary molecules
        for i,secmol in enumerate(meas['head']['secmol']):
            plotmarginalsled(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,n_comp,nicenames,n_mol,
                title=title+' '+secmol,colors=colors,simplify=simplify,
                useind=5+i,mol=secmol)            