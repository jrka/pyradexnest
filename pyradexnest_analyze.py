#
# Be able to run this given just a directory to access with finished run inside.
# import pymultinest
# execfile('/arlstorage/home/student/kamenetz/ftssurvey/pymultinest/plotjrk.py')
#
# NO, NEW:
# python plotjrk.py 'title'
#
# 7/17/13: There was an error in get_best_fit.  Fixed, also, use that
#    for maxima in table and vertical lines.f
# 7/18/13: Implemented upper limits.
# 7/19/13: Now "secondary parameters" are output at the end of the cube.
#    Remove "sec" options for binning results.
#  8/2/13: Added mass1 (and mass2) to table output.
#  8/9/13: Define generic plotting routines.
# 8/12/13: Use analyze_tools.py
# 8/20/13: Input title and basedir as sysargs.
#
import pymultinest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import os
import cPickle as pickle
import pyradex as pyradex
from multiplot import multipanel
from astropy.table import Table, Column
import numpy
import sys
import astropy.units as units
from analyze_tools import *

#################################################################################

n_dims=8
if n_dims==8:
    n_sec=[6,3]
else:
    n_sec=3
#n_sec=3*(n_dims)/4 # now we also have ratio.
n_params =n_dims + sum(n_sec)
norm1=True

meas=pickle.load(open("measdata.pkl","rb"))
lw=numpy.log10(meas['head']['lw'])

a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()
data= a.get_data()
datsep=post_sep(a.post_file)  # Divide the "data" up by mode.
datsep['all']=data
bestfit=a.get_best_fit()
cube=bestfit['parameters'] # The best fit is not the same as the mode, cube=s['modes'][0]['maximum']
nmodes=len(s['modes'])

# Stupid to have to input n_params into Analyzer.  Overwrite.
n_params=len(s['modes'][0]['mean'])

if n_params>7:
    parameters = ["h2den1","tkin1","cdmol1","ff1","h2den2","tkin2","cdmol2","ff2","lum1","press1","bacd1","lum2","press2","bacd2","lumratio","pressratio","bacdratio"] 
    n_dims=8
    n_sec=[6,3]
    n_params =n_dims + sum(n_sec)
    add=[0,0,lw,0, 0,0,lw,0, lw,0,lw, lw,0,lw, 0,0,0]
    mult=[1,1,1,1, 1,1,1,1, 1,1,1, 1,1,1, 1,1,1]
    colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k']
    plotinds=[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16]]
else:
    parameters = ["h2den1","tkin1","cdmol1","ff1","lum1","press1","bacd1"] 
    n_dims=4
    n_sec=3*(n_dims)/4
    n_params =n_dims + n_sec
    add=[0,0,lw,0, lw,0,lw,]
    mult=[1,1,1,1, 1,1,1]
    colors=['b','b','b','b','b','b','b']
    plotinds=[[0,1,2,3],[4,5,6]]
    
nicenames=[r'log(n$_{H2}$ [cm$^{-3}$])',r'log(T$_{kin}$ [K])',r'log(N$_{CO}$ [cm$^{-2}$)',r'log($\Phi$)',
           r'log(L[erg s$^{-1}$])',r'log(Pressure [K cm$^{-2}$])',r'log(<N$_{CO}$> [cm$^{-2}$]',
           r'log(Ratio L$_{warm}$/L$_{cold}$)',r'log(Ratio P$_{warm}$/P$_{cold}$)',r'log(Ratio <N>$_{warm}$/<N>$_{cold}$)']
#nicenames_marg=[r'log(n$_{H2}$)',r'log(T$_{kin}$)',r'log(Pressure)',r'log(N$_{CO}$)',r'log($\Phi$)',r'log(BACD)']
#xrange_marg=[[2,6.5],[0.7,3.5],[2.7,10],[12,19]+lw,[-3,0],[9,19]+lw]

#xrange=[[2,6],[0.7,3.7],[15,18],[-3,0]] # matches M82
xrange=[[2,6.5],[0.7,3.5],[12,19]+lw,[-3,0]] # from cube
modecolors=['g','m','y','c','k','r','b']

######################################

# If a binned pickle already exists and is more 
#   recent than chains, use it.
# Otherwise, do all the binning and save it to a pickle.

distfile='distributions.pkl'
dists=get_dists(distfile,s,datsep,grid_points=40)

######################################
# Table.

table=maketables(s,n_params,parameters,cube,add,mult,title=title,addmass=meas['addmass'],n_dims=n_dims)
if n_dims>7:
    modemed=table['Mode Mean'][0:17]-add
    modemax=table['Mode Maximum'][0:17]-add
else:
    modemed=table['Mode Mean'][0:7]-add
    modemax=table['Mode Maximum'][0:7]-add
pickle.dump(table, open('table.pkl', "wb") )


######################################
# Plots

doplot=True
# For now....
if doplot:
    plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title=title,norm1=norm1,colors=colors)    
    plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title=title,norm1=norm1,colors=colors,meas=meas)
    plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames)
    plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames)
    plotsled(meas,cube,n_params,modemed,modemax,title='')
