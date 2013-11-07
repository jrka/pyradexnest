import pymultinest
import math
import os, threading, subprocess
import numpy
from astropy.table import Table,join
import pyradex as pyradex
import cPickle as pickle
import sys
import time
# Run this in the output directory!  Make "chains" before running with mpi.
if not os.path.exists("chains"): os.mkdir("chains")

# read in measdata ######################################################################
def measdata_pickle(measdatafile):
    from astropy import constants as const
    from astropy import units as units
    import astropy.coordinates.distances as dist
    import astropy.io.ascii as ascii
    import pprint
    
    jytocgs=1.0e-23
    top=numpy.genfromtxt(measdatafile,delimiter="\n",comments="#")
    head={'omega_s':top[0],'z':top[1],'mag':top[2],'lw':top[3],'abundance':top[4],'dynmass':top[5],'mol_weight':top[6],'length':top[7]}
    
    data=ascii.read(measdatafile,delimiter='\s',comment='#',guess=False,Reader=ascii.NoHeader,data_start=8,
    names=['mol','J_up','J_lo','freq','FLUX_Jykms','measerr','calerr'])
    # Convert to K from Jy km/s (including dividing by linewidth), and ditch the astropy table.
    flux=numpy.array( (const.c.to('cm/s')/(data['freq']*1e9))**2/(2.0*const.k_B.to('erg/K')))
    flux*=(data['FLUX_Jykms']*jytocgs)/head['omega_s'] * (1.0+head['z'])/head['mag']
    flux/=head['lw']
    
    sigma=numpy.array( (const.c.to('cm/s')/(data['freq']*1e9))**2/(2.0*const.k_B.to('erg/K')))
    sigma*=(data['measerr']*jytocgs)/head['omega_s'] * (1.0+head['z'])/head['mag']
    sigma/=head['lw']
    
    sigma[flux >0]=numpy.sqrt(sigma[flux>0]**2+(flux[flux>0]*data['calerr'][flux>0]/100.0)**2)
    sigma[flux==0]*=(1.0+data['calerr'][flux==0]/100.0)

    jup=numpy.array(data['J_up'])
    
    # Require at least 3 unique lines to model.
    if len(numpy.unique(jup)) < 3: raise Exception("Not enough lines to model.")
           
    # Also calculate the "prior cuts" for mass and length (in "cube" units), add that to the meas structure.
    # Based off of rtl3_multi_make_prior_cuts.pro
    if head['dynmass'] != -1:
        angdist=dist.Distance(z=head['z']).pc/(1.0+head['z'])**2
        s_area=head['omega_s']*(angdist)**2/(1.0+head['z'])**2 # pc^2
        NCO_dv_cut = (head['dynmass']*units.solMass.to('kg'))/(2.0*units.M_p.to('kg')*head['mol_weight'])
        NCO_dv_cut*= head['abundance']/(s_area*units.pc.to('cm')**2)
        NCO_dv_cut/=head['lw'] # Must check it.
        NCO_dv_cut=numpy.log10(NCO_dv_cut)
    else:
        NCO_dv_cut=100
    
    if head['length'] != -1:
        gallength=head['length']*units.pc.to('cm')*head['abundance'] #length*cm_per_pc*abundance, cm
        gallength = numpy.log10(gallength)
    else:
        gallength=100

    # For use in later calculation, how to get from BACD/dv to mass/dv:
    addmass=numpy.log10(s_area*units.pc.to('cm')**2)+numpy.log10(2.0*units.M_p.to('kg')*units.kg.to('solMass')*head['mol_weight'])-numpy.log10(head['abundance'])

    meas = {'J_up': jup, 'flux':flux, 'sigma':sigma, 
            'masscut':NCO_dv_cut, 'lengthcut': gallength, 'head':head, 'areacm':numpy.log10(s_area*units.pc.to('cm')**2),
            'sigmacut':3.0, 'taulimit':[-0.9,100.0], 'addmass':addmass}
    pprint.pprint(meas)
    
    pickle.dump(meas, open("measdata.pkl", "wb") )

# prior function ########################################################################
def myprior(cube, ndim, nparams):
    cube[0]=cube[0]*4.5+2  # h2den1  2 to 6.5
    # Temperature 1 range will depend on if this is 1 component (full temp range) or 2 component (limited temp range)
    cube[2]=cube[2]*7+12   # cdmol1  12 to 19
    cube[3]=cube[3]*3-3    # ff1     -3 to 0
    if ndim>4:
        cube[1]=cube[1]*2.2+0.5# tkin1   0.5 to 2.3 = 3.16 to 502 K.
        cube[4]=cube[4]*4.5+2  # h2den2  2 to 6.5
        cube[5]=cube[5]*1.5+2  # tkin2   2 to 3.5 = 100 to 3162 K
        cube[6]=cube[6]*7+12   # cdmol2  12 to 19
        cube[7]=cube[7]*3-3    # ff2     -3 to 0
    else:
        cube[1]=cube[1]*3+0.5

# loglikelihood function ################################################################
def myloglike(cube, ndim, nparams):
    # Calculate the log(likelihood).  Load in the measdata.pkl for our data,
    # and use pyradex to get our model.
    # If we violate any priors, limits, or have calculation errors, return -2e100 as the likelihood.

    meas=pickle.load(open("measdata.pkl","rb"))
    taumin=meas['taulimit'][0]#-0.9
    taumax=meas['taulimit'][1]#100.0
    sigmacut=meas['sigmacut']#3.0
    
    import warnings
    
    # First, test that the cube parameters do not violate the mass or length priors.
    if ndim>4:
        if cube[6]+cube[7] > cube[2]+cube[3] or \
           cube[1] > cube[5] or \
           math.log10(math.pow(10,cube[2]+cube[3])+math.pow(10,cube[6]+cube[7])) > meas['masscut'] or \
           cube[2]-cube[0]-0.5*cube[3] > meas['lengthcut'] or \
           cube[6]-cube[4]-0.5*cube[7] > meas['lengthcut']:    
               return -2e100
    else:
        if cube[2]+cube[3] > meas['masscut'] or cube[2]-cube[0]-0.5*cube[3] > meas['lengthcut']:
            return -2e100

    # Call RADEX for the first component.
    try:
        dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=math.pow(10,cube[1]), column=math.pow(10,cube[2]), 
                          collider_densities={'H2':math.pow(10,cube[0])},
                          tbg=2.73, species='co', velocity_gradient=1.0, debug=True)
        dat['FLUX_Kkms']*=math.pow(10,cube[3])
    except:
        return -2e100
    
    # If applicable, call RADEX for the second component and find their sum.
    # Either way, check if any optical depths are outside of our limits.
    if ndim>4:
        try:
            dat2=pyradex.pyradex(flow=1, fhigh=1600,
                               tkin=math.pow(10,cube[5]), column_density=math.pow(10,cube[6]), 
                               collider_densities={'H2':math.pow(10,cube[4])},
                               tbg=2.73, molecule='co', velocity_gradient=1.0, debug=False)
            dat2['FLUX_Kkms']*=math.pow(10,cube[7])
            newdat=dat['FLUX_Kkms']+dat2['FLUX_Kkms'] # We want to compare the SUM of the components to our data.
            tauok=numpy.all([dat['TAU']<taumax,dat['TAU']>taumin,dat2['TAU']<taumax,dat2['TAU']>taumin],axis=0)
        except:
            return -2e100
    else:
        newdat=dat['FLUX_Kkms']
        tauok=numpy.all([dat['TAU']<taumax,dat['TAU']>taumin],axis=0)
    
    # Check that we have at least one line with optical depth in allowable limits.
    if not numpy.any(tauok):
        return -2e100
        
    # Check that we don't violate ANY line flux upper limits.
    ulok=newdat[meas['flux']==0] < meas['sigma'][meas['flux']==0]*sigmacut
    if not numpy.all(ulok):
        return -2e100
    
    # We've made it!  Calculate likelihood!
    nmeas=len(meas['sigma'])
    loglike=-nmeas*0.5*math.log(2.0*math.pi)
    for i in range(nmeas):
        try:
            if tauok[i] and meas['flux'][i] > 0:
                loglike=loglike-math.log(meas['sigma'][i])
                loglike=loglike-0.5*(math.pow((meas['flux'][i]-newdat[dat['J_up'] == meas['J_up'][i]])/meas['sigma'][i],2))
        except:
            warnings.warn('An error occured with J_up = '+str(meas['J_up'][i])+', will be ignored.')
            loglike=loglike
            
    # Record the luminosity, pressure, and beam-averaged column density for binning later.
    cube[ndim]=dat['LogLmol']+meas['areacm']+cube[2]+cube[3] #0 
    cube[ndim+1]=cube[0]+cube[1]
    cube[ndim+2]=cube[2]+cube[3]
    # If we have 2 components, also records those L, P, and BACD, as well
    # as ratios of the warm to cold (note these are in log, so they are differenced).
    if ndim>4:
        cube[ndim+3]=dat2['LogLmol']+meas['areacm']+cube[6]+cube[7] #0
        cube[ndim+4]=cube[4]+cube[5]
        cube[ndim+5]=cube[6]+cube[7]
        # Ratios of WARM to COLD
        cube[ndim+6]=cube[ndim+3]-cube[ndim]
        cube[ndim+7]=cube[ndim+4]-cube[ndim+1]
        cube[ndim+8]=cube[ndim+5]-cube[ndim+2]
    
    return loglike

######################################################################################
# main function ######################################################################

# USER SETTINGS HERE, n_dims=4 or 8
n_dims=4

# The file to read in
cwd=os.getcwd()
split=cwd.rpartition('/') # base this entirely on the directory you are in.
measdatafile='measdata.txt'

# number of dimensions our problem has. can do 4 for 1 component, or 8 for 2 components.

if n_dims==4:
    parameters=["h2den1","tkin1","cdmol1","ff1","lum1","press1","bacd1"]
elif n_dims==8:
    parameters = ["h2den1","tkin1","cdmol1","ff1","h2den2","tkin2","cdmol2","ff2",
              "lum1","press1","bacd1","lum2","press2","bacd2",
              "lumratio","pressratio","bacdratio"] 
else: 
    print 'Follow directions!  Using 4 parameters.'
    n_dims=4
    parameters=["h2den1","tkin1","cdmol1","ff1","lum1","press1","bacd1"]

n_params = len(parameters)
measdata_pickle(measdatafile)

# if you got this far, test it out if you want:
testcube=[0.5,0.5,0.5,0.5]

# we want to see some output while it is running.  Since I'm not using show, are tehse necessary?
progress = pymultinest.ProgressPlotter(n_params = n_params); progress.start()
#threading.Timer(4, show, ["chains/1-phys_live.points.pdf"]).start() # delayed opening # don't show 
# run MultiNest
pymultinest.run(myloglike, myprior, n_dims, n_params=n_params,resume = False, verbose = True, sampling_efficiency = 0.8,
                n_live_points = 500)
# ok, done. Stop our progress watcher
progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()

import json
json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
print
print "-" * 30, 'ANALYSIS', "-" * 30
print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )


############# Don't use below anymore?
import matplotlib.pyplot as plt

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0)
for i in range(n_params):
	plt.subplot(n_params, n_params, n_params * i + i + 1)
	p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
	plt.ylabel("Probability")
	plt.xlabel(parameters[i])
	
	for j in range(i):
		plt.subplot(n_params, n_params, n_params * j + i + 1)
		#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
		p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
		plt.xlabel(parameters[i])
		plt.ylabel(parameters[j])

plt.savefig("marginals_multinest.pdf") #, bbox_inches='tight')
#show("marginals_multinest.pdf")
for i in range(n_params):
	outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
	p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
	plt.ylabel("Probability")
	plt.xlabel(parameters[i])
	plt.savefig(outfile, format='pdf', bbox_inches='tight')
	plt.close()
	
	outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
	p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
	plt.ylabel("Cumulative probability")
	plt.xlabel(parameters[i])
	plt.savefig(outfile, format='pdf', bbox_inches='tight')
	plt.close()

print "take a look at the pdf files in chains/" 
