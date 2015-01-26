import pymultinest
import os, threading, subprocess
import numpy as np
#from astropy.table import Table,join
import pyradex as pyradex
import cPickle as pickle
from config import *
import matplotlib.pyplot as plt

def show(filepath): 
	""" open the output (pdf) file for the user """
	if os.name == 'mac': subprocess.call(('open', filepath))
	elif os.name == 'nt': os.startfile(filepath)
	elif os.name == 'posix': subprocess.call(('open', filepath)) # JRK changed 'xdg-open' to 'open'

# Run this in the output directory!  Make "chains" before running with mpi.
if not os.path.exists("chains"): os.mkdir("chains")

# read in measdata ######################################################################
def measdata_pickle(measdatafile,sled_to_j=False):
    from astropy import constants as const
    from astropy import units as units
    import astropy.coordinates.distances as dist
    import astropy.io.ascii as ascii
    
    jytocgs=1.0e-23
    top=np.genfromtxt(measdatafile,delimiter="\n",comments="#")
    head={'omega_s':top[0],'z':top[1],'mag':top[2],'lw':top[3],'abundance':top[4],'dynmass':top[5],'mol_weight':top[6],'length':top[7]}
    
    data=ascii.read(measdatafile,delimiter='\s',comment='#',guess=False,Reader=ascii.NoHeader,data_start=8,
    names=['mol','J_up','J_lo','freq','FLUX_Jykms','measerr','calerr'])
    # Convert to K from Jy km/s (including dividing by linewidth), and ditch the astropy table.
    flux=np.array( (const.c.to('cm/s')/(data['freq']*1e9))**2/(2.0*const.k_B.to('erg/K')))
    flux*=(data['FLUX_Jykms']*jytocgs)/head['omega_s'] * (1.0+head['z'])/head['mag']
    flux/=head['lw']
    
    sigma=np.array( (const.c.to('cm/s')/(data['freq']*1e9))**2/(2.0*const.k_B.to('erg/K')))
    sigma*=(data['measerr']*jytocgs)/head['omega_s'] * (1.0+head['z'])/head['mag']
    sigma/=head['lw']
    
    sigma[flux >0]=np.sqrt(sigma[flux>0]**2+(flux[flux>0]*data['calerr'][flux>0]/100.0)**2)
    sigma[flux==0]*=(1.0+data['calerr'][flux==0]/100.0)

    jup=np.array(data['J_up'])
    
    # Require at least 3 unique lines to model.
    if len(np.unique(jup)) < 3: raise Exception("Not enough lines to model.")
           
    # Also calculate the "prior cuts" for mass and length (in "cube" units), add that to the meas structure.
    # Based off of rtl3_multi_make_prior_cuts.pro
    if head['dynmass'] != -1:
        angdist=dist.Distance(z=head['z']).pc/(1.0+head['z'])**2
        s_area=head['omega_s']*(angdist)**2/(1.0+head['z'])**2 # pc^2
        NCO_dv_cut = (head['dynmass']*units.solMass.to('kg'))/(2.0*units.M_p.to('kg')*head['mol_weight'])
        NCO_dv_cut*= head['abundance']/(s_area*units.pc.to('cm')**2)
        NCO_dv_cut/=head['lw'] # Must check it.
        NCO_dv_cut=np.log10(NCO_dv_cut)
    else:
        NCO_dv_cut=100
    
    if head['length'] != -1:
        gallength=head['length']*units.pc.to('cm')*head['abundance'] #length*cm_per_pc*abundance, cm
        gallength = np.log10(gallength)
    else:
        gallength=100

    # For use in later calculation, how to get from BACD/dv to mass/dv:
    addmass=np.log10(s_area*units.pc.to('cm')**2)+np.log10(2.0*units.M_p.to('kg')*units.kg.to('solMass')*head['mol_weight'])-np.log10(head['abundance'])

    meas = {'J_up': jup, 'flux':flux, 'sigma':sigma, 
            'masscut':NCO_dv_cut, 'lengthcut': gallength, 'head':head, 'areacm':np.log10(s_area*units.pc.to('cm')**2),
            'sigmacut':3.0, 'taulimit':[-0.9,100.0], 'addmass':addmass, 'sled_to_j':sled_to_j}
    print meas
    
    pickle.dump(meas, open("measdata.pkl", "wb") )

# prior function ########################################################################
# The prior function is in the config file.  This allows you to choose different ranges
# for parameters with different runs.

# loglikelihood function ################################################################
def myloglike(cube, ndim, nparams):
    import warnings
    # Calculate the log(likelihood).  Load in the measdata.pkl for our data,
    # and use pyradex to get our model.
    # If we violate any priors, limits, or have calculation errors, return -2e100 as the likelihood.
    meas=pickle.load(open("measdata.pkl","rb"))
    taumin=meas['taulimit'][0]
    taumax=meas['taulimit'][1]
    sigmacut=meas['sigmacut']
    
    # First, test that the cube parameters do not violate the mass or length priors.
    #  Warm mass cannot be greater than cold mass.
    #  Cold temperature cannot be less than warm temperature
    #  Total mass cannot be greater than dynamical mass limit
    #  Neither length can be greater than the length limits
    if ndim>4:
        if cube[6]+cube[7] > cube[2]+cube[3] or \
           cube[1] > cube[5] or \
           np.log10(np.power(10,cube[2]+cube[3])+np.power(10,cube[6]+cube[7])) > meas['masscut'] or \
           cube[2]-cube[0]-0.5*cube[3] > meas['lengthcut'] or \
           cube[6]-cube[4]-0.5*cube[7] > meas['lengthcut']:    
               return -2e100
    else:
        if cube[2]+cube[3] > meas['masscut'] or cube[2]-cube[0]-0.5*cube[3] > meas['lengthcut']:
            return -2e100

    # Call RADEX for the first component.
    try:
        dat1=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=2.73, species='co', velocity_gradient=1.0, debug=False,
                          return_dict=True)
        # dat['J_up'] returned as strings; this is fine for CO...
        jup1=np.array(map(float,dat1['J_up']))
        model1=np.array(map(float,dat1['FLUX_Kkms']))*np.power(10,cube[3])
        tau1=np.array(map(float,dat1['TAU']))
        
        # Check for convergence
        if dat1['niter']==['****']: return -2e100
        
        # At this time it is too slow to use this.
        #R = pyradex.Radex(collider_densities={'h2':np.power(10,cube[0])}, 
        #     temperature=np.power(10,cube[1]), column=np.power(10,cube[2]),
        #     tbackground=2.73,species='co',deltav=1.0,debug=False)
        #niter=R.run_radex(validate_colliders=False)
        #model1=1.064575*R.T_B*np.power(10,cube[3]) # Integrating over velocity, and Filling Factor
        #model1=model1.value # Hatred of units in my way
        #tau1=R.tau
        #jup1=R.upperlevelindex        
        #cube[ndim]=np.sum(R.source_brightness_beta) # luminosity; not correct units yet.
        
    except:
        return -2e100

    # Which indices are lines that we are concerned with?
    juse=np.in1d(jup1.ravel(),meas['J_up']).reshape(jup1.shape)
    jmeas=np.in1d(jup1.ravel(),meas['J_up'][meas['flux']!=0]).reshape(jup1.shape)
    
    # If applicable, call RADEX for the second component and find their sum.
    # Either way, check if any optical depths are outside of our limits.
    if ndim>4:
        try:
            dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
                               temperature=np.power(10,cube[5]), column=np.power(10,cube[6]), 
                               collider_densities={'H2':np.power(10,cube[4])},
                               tbg=2.73, species='co', velocity_gradient=1.0, debug=False,
                               return_dict=True)
            jup2=np.array(map(float,dat2['J_up']))
            model2=np.array(map(float,dat2['FLUX_Kkms']))*np.power(10,cube[7])
            tau2=np.array(map(float,dat2['TAU']))                            

            # Check for convergence
            if dat2['niter']==['****']: return -2e100
                               
            modelt=model1+model2# We want to compare the SUM of the components to our data.
            tauok=np.all([tau1<taumax,tau1>taumin,tau2<taumax,tau2>taumin],axis=0)
            
            #R.temperature=np.power(10,cube[5])
            #R.density=np.power(10,cube[4])
            #R.column=np.power(10,cube[6])
            #niter=R.run_radex(validate_colliders=False)
            
            #model2=1.064575*R.T_B*np.power(10,cube[3])# Integrating over velocity, and Filling Factor
            #model2=model2.value
            #tau2=R.tau
            #jup2=R.upperlevelindex
            #if not np.array_equal(jup1,jup2): 
            #    warnings.warn('J_up arrays NOT equal from multiple RADEX calls!')
            #    return -2e100
            #modelt=model1+model2
            #tauok=np.all([tau1<taumax,tau1>taumin,tau2<taumax,tau2>taumin],axis=0)
        except:
            return -2e100
    else:
        modelt=model1# total model
        tauok=np.all([tau1<taumax,tau1>taumin],axis=0)
    
    # Check that we have at least one line with optical depth in allowable limits.
    if not np.any(tauok[jmeas]):
        return -2e100
        
    # Check that we don't violate ANY line flux upper limits.
    ul=np.where(meas['flux']==0)[0]
    for i in ul:
        ulok=modelt[jup1==meas['J_up'][i]] < meas['sigma'][i]*sigmacut
        if not ulok: return -2e100
    
    # We've made it!  Calculate likelihood!
    nmeas=len(meas['sigma'])
    loglike=-nmeas*0.5*np.log(2.0*np.pi)
    for i in range(nmeas):
        try:
            j_ind=np.where(jup1==meas['J_up'][i])[0]
            if tauok[j_ind] and meas['flux'][i] > 0:
                loglike=loglike-np.log(meas['sigma'][i])
                loglike=loglike-0.5*(np.power((meas['flux'][i]-modelt[j_ind])/meas['sigma'][i],2))
        except:
            warnings.warn('An error occured with j_up = '+str(meas['J_up'][i])+', will be ignored.')
            loglike=loglike
            
    # Record the luminosity, pressure, and beam-averaged column density for binning later.
    #cube[ndim]=dat['LogLmol']+meas['areacm']+cube[2]+cube[3] #0  ##### NEED THIS LATER
    cube[ndim]=1.0  # FILL IN LATER.
    cube[ndim+1]=cube[0]+cube[1]
    cube[ndim+2]=cube[2]+cube[3]

    # Don't include sled likelihoods with bad tau - cannot NaN them :( ?
    
    # Watch out, in the rare chance of a ridiculous RADEX value of something E+99, 
    #  MultiNest will print 0.XYZ+100 instead of X.YZE+99, and will not be read as float.
    #  You've not used this value in your likelihood because tau will be wild as well.
    model1[model1>9e98]=9e98
    if ndim>4: model2[model2>9e98]=9e98

    # If we have 2 components, also records those L, P, and BACD, as well
    # as ratios of the warm to cold (note these are in log, so they are differenced).
    if ndim>4:
        #cube[ndim+3]=dat2['LogLmol']+meas['areacm']+cube[6]+cube[7] #0  #### NEED THIS LATER
        cube[ndim+3]=1.0  # FILL IN LATER
        cube[ndim+4]=cube[4]+cube[5]
        cube[ndim+5]=cube[6]+cube[7]
        # Ratios of WARM to COLD
        cube[ndim+6]=cube[ndim+3]-cube[ndim]
        cube[ndim+7]=cube[ndim+4]-cube[ndim+1]
        cube[ndim+8]=cube[ndim+5]-cube[ndim+2]
        # SLED likelihoods, unhappy with slice indices...
        for i,l in enumerate(model1[0:meas['sled_to_j']]): cube[ndim+9+i]=l
        for i,l in enumerate(model2[0:meas['sled_to_j']]): cube[ndim+9+meas['sled_to_j']+i]=l
        #cube[ndim+9:]=np.append(model1[0:meas['sled_to_j']].value,model2[0:meas['sled_to_j']])
    else:
        for i,l in enumerate(model1[0:meas['sled_to_j']]): cube[ndim+3+i]=l 
        #cube[ndim+3:ndim+3+meas['sled_to_j']]=model1[0:meas['sled_to_j']].value
    
    return loglike

######################################################################################
# main function ######################################################################

# User settings loaded into config.py already

# The file to read in
cwd=os.getcwd()
split=cwd.rpartition('/')
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

if sled_to_j and n_dims==4: map(parameters.append,["k"+str(i) for i in range(1,sled_to_j+1)])
if sled_to_j and n_dims==8: 
    map(parameters.append,["k"+str(i)+"c" for i in range(1,sled_to_j+1)]) # Cold
    map(parameters.append,["k"+str(i)+"w" for i in range(1,sled_to_j+1)]) # Warm
n_params = len(parameters)

# Created our measured data pickle.
measdata_pickle(measdatafile,sled_to_j=sled_to_j)

# Before starting, record the parameter limits, min and max of prior cube.
# Outdated; now this is recorded in config.py.  But in case you've changed it...
pmin=np.zeros(n_dims)
pmax=np.zeros(n_dims)+1
myprior(pmin,n_dims,n_params)
myprior(pmax,n_dims,n_params)
np.savetxt('prior_cube.txt',[pmin,pmax],fmt='%+.3e')

# Still avoiding the progress plotter.
#progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/1-'); progress.start()
#threading.Timer(2, show, ["chains/1-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(myloglike, myprior, n_dims, n_params=n_params, importance_nested_sampling = False, resume = False, 
                verbose = True, sampling_efficiency = 'model', n_live_points = 500, 
                outputfiles_basename='chains/1-',init_MPI=False)
# ok, done. Stop our progress watcher
#progress.stop()

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params)
s = a.get_stats()

import json
json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
print
print "-" * 30, 'ANALYSIS', "-" * 30
print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )
print "Run pyradexnest_analyze.py"

# Keep the basic plotting here for now, until mine is ready.  Actually, don't do 
# the first one, there are too many parameters and the PDF will explode.
p = pymultinest.PlotMarginalModes(a)
#plt.figure(figsize=(5*n_params, 5*n_params))
##plt.subplots_adjust(wspace=0, hspace=0)
#for i in range(n_params):
#       plt.subplot(n_params, n_params, n_params * i + i + 1)
#       p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
#       plt.ylabel("Probability")
#       plt.xlabel(parameters[i])
#       
#       for j in range(i):
#               plt.subplot(n_params, n_params, n_params * j + i + 1)
#               #plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
#               p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
#               plt.xlabel(parameters[i])
#               plt.ylabel(parameters[j])
#plt.savefig("marginals_multinest.pdf") #, bbox_inches='tight')

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