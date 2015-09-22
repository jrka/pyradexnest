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

# JRK 8/13/15: Was missing title in calls to plotconditional and plotconditional2.
# JRK 9/22/15: Don't replicate the import calls from pyradexnest_analyze_tools

#################################################################################

#### LOAD THE STATS
title=os.getcwd()
title=title.split('/')[-1]

# Total number of parameters: Dimensions, "Secondary Parameters", SLED Likelihoods
if n_dims==8:
    n_sec=[6,3]
    n_sled=2*sled_to_j
else:
    n_sec=[3]
    n_sled=sled_to_j
n_params =n_dims + np.sum(n_sec) + n_sled

meas=pickle.load(open("measdata.pkl","rb"))
lw=np.log10(meas['head']['lw'])

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
[parameters,add,mult,colors,plotinds,sledcolors]=define_plotting(n_dims,n_sec,sled_to_j,lw)
modecolors=['g','m','y','c','k','r','b']
    
nicenames=[r'log(n$_{H2}$ [cm$^{-3}$])',r'log(T$_{kin}$ [K])',r'log(N$_{CO}$ [cm$^{-2}$])',r'log($\Phi$)',
           r'log(L[erg s$^{-1}$])',r'log(Pressure [K cm$^{-2}$])',r'log(<N$_{CO}$> [cm$^{-2}$]',
           r'log(Ratio L$_{warm}$/L$_{cold}$)',r'log(Ratio P$_{warm}$/P$_{cold}$)',r'log(Ratio <N>$_{warm}$/<N>$_{cold}$)']
if sled_to_j:
    for x in range(sled_to_j): nicenames.append(r'Flux J='+str(x+1)+'-'+str(x)+' [K]')

# Determine plot xrange from the prior limits.
xrange=np.ndarray([n_dims,2])
xrange[:,0]=0
xrange[:,1]=1
myprior(xrange[:,0],n_dims,n_params)
myprior(xrange[:,1],n_dims,n_params)

# Squash it down if we have 2 components.
if n_dims==8:
    for i in range(4):
        xrange[i,0]=min(xrange[i,0],xrange[i+4,0])
        xrange[i,1]=max(xrange[i,1],xrange[i+4,1])
    xrange=xrange[0:4,:]

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

######################################
# Table.

table=maketables(s,n_params,parameters,cube,add,mult,title=title,addmass=meas['addmass'],n_dims=n_dims)
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
    plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
        xr=xrange,title=title,norm1=norm1,colors=colors)    
    plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
        xr=xrange,title=title,norm1=norm1,colors=colors,meas=meas)
    
    plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title=title)
    plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title=title)
    
    plotsled(meas,cube,n_params,n_dims,modemed,modemax,modesig,plotinds,title='',sled_to_j=sled_to_j)
    
    if sled_to_j: plotmarginalsled(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title=title,colors=colors)