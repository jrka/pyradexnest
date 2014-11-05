import pymultinest
import os, threading, subprocess
import numpy as np
#from astropy.table import Table,join
import pyradex as pyradex
import pyradexv3 as pyradexv3
import cPickle as pickle
from config import *

# Run this in the output directory!  Make "chains" before running with mpi.
if not os.path.exists("chains"): os.mkdir("chains")

# read in measdata ######################################################################
@profile
def measdata_pickle(measdatafile,sled_to_j=False):
    from astropy import constants as const
    from astropy import units as units
    import astropy.coordinates.distances as dist
    import astropy.io.ascii as ascii
    import pprint
    
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
    pprint.pprint(meas)
    
    pickle.dump(meas, open("measdata.pkl", "wb") )

# prior function ########################################################################
# The prior function is in the config file.  This allows you to choose different ranges
# for parameters with different runs.

# loglikelihood function ################################################################
@profile
def myloglike(cube, ndim, nparams):
    import warnings
    # Calculate the log(likelihood).  Load in the measdata.pkl for our data,
    # and use pyradex to get our model.
    # If we violate any priors, limits, or have calculation errors, return -2e100 as the likelihood.
    meas=pickle.load(open("measdata.pkl","rb"))
    taumin=meas['taulimit'][0]#-0.9
    taumax=meas['taulimit'][1]#100.0
    sigmacut=meas['sigmacut']#3.0
    
    # First, test that the cube parameters do not violate the mass or length priors.
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
        #dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
        #                  temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), 
        #                  collider_densities={'H2':np.power(10,cube[0])},
        #                  tbg=2.73, species='co', velocity_gradient=1.0, debug=False)
        #dat['flux_kkms']*=np.power(10,cube[3])
        
        R = pyradex.Radex(collider_densities={'h2':np.power(10,cube[0])}, 
             temperature=np.power(10,cube[1]), column=np.power(10,cube[2]),
             tbackground=2.73,species='co',deltav=1.0,debug=False)
        niter=R.run_radex(validate_colliders=False)
        
        
        
        
        model1=1.064575*R.T_B*np.power(10,cube[3]) # Integrating over velocity, and Filling Factor
        model1=model1.value # Hatred of units in my way
        tau1=R.tau
        jup1=R.upperlevelindex
        
        # Which indices are lines that we are concerned with?
        juse=np.in1d(jup1.ravel(),meas['J_up']).reshape(jup1.shape)
        
        #dat=R(collider_densities={'h2':np.power(10,cube[0])}, temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), tbackground=2.73,species='co',deltav=1.0)
        
        #cube[ndim]=np.sum(R.source_brightness_beta) # luminosity; not correct units yet.
        # Need R.surface_brightness_beta() for luminosity.
        
    except:
        return -2e100
    
    # If applicable, call RADEX for the second component and find their sum.
    # Either way, check if any optical depths are outside of our limits.
    if ndim>4:
        try:
            #dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
            #                   temperature=np.power(10,cube[5]), column=np.power(10,cube[6]), 
            #                   collider_densities={'H2':np.power(10,cube[4])},
            #                   tbg=2.73, species='co', velocity_gradient=1.0, debug=False)
            #dat2['flux_kkms']*=np.power(10,cube[7])
            #newdat=dat['flux_kkms']+dat2['flux_kkms'] # We want to compare the SUM of the components to our data.
            #tauok=np.all([dat['tau']<taumax,dat['tau']>taumin,dat2['tau']<taumax,dat2['tau']>taumin],axis=0)
            
            #R = pyradex.Radex(collider_densities={'h2':np.power(10,cube[4])}, 
            # temperature=np.power(10,cube[5]), column=np.power(10,cube[6]),
            # tbackground=2.73,species='co',deltav=1.0,debug=False)
            #niter=R.run_radex(validate_colliders=False)
            
            R.density=np.power(10,cube[4])
            R.temperature=100 #np.power(10,cube[5])
            R.column=np.power(10,cube[6])
            niter=R.run_radex(validate_colliders=False)
            
            model2=1.064575*R.T_B*np.power(10,cube[3])# Integrating over velocity, and Filling Factor
            model2=model2.value
            tau2=R.tau
            jup2=R.upperlevelindex
            if not np.array_equal(jup1,jup2): 
                warnings.warn('J_up arrays NOT equal from multiple RADEX calls!')
                return -2e100

            
            modelt=model1+model2
            tauok=np.all([tau1<taumax,tau1>taumin,tau2<taumax,tau2>taumin],axis=0)
        except:
            return -2e100
    else:
        modelt=model1 # total model
        tauok=np.all([tau1<taumax,tau1>taumin],axis=0)
    
    # Check that we have at least one line with optical depth in allowable limits.
    if not np.any(tauok[juse]):
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

@profile
def myloglike_v3(cube, ndim, nparams):
    import math
    import numpy as numpy
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
        dat=pyradexv3.radex(flow=1, fhigh=1600,
                          tkin=math.pow(10,cube[1]), column_density=math.pow(10,cube[2]), 
                          collider_densities={'H2':math.pow(10,cube[0])},
                          tbg=2.73, molecule='co', velocity_gradient=1.0, debug=False)
        dat['FLUX_Kkms']*=math.pow(10,cube[3])
    except:
        return -2e100
    
    # If applicable, call RADEX for the second component and find their sum.
    # Either way, check if any optical depths are outside of our limits.
    if ndim>4:
        try:
            dat2=pyradexv3.radex(flow=1, fhigh=1600,
                               tkin=math.pow(10,cube[5]), column_density=math.pow(10,cube[6]), 
                               collider_densities={'H2':math.pow(10,cube[4])},
                               tbg=2.73, molecule='co', velocity_gradient=1.0, debug=True)
            dat2['FLUX_Kkms']*=math.pow(10,cube[7])
        except:
            print 'Call to pyradexv3 for dat2 failed'
            return -2e100
        newdat=dat['FLUX_Kkms']+dat2['FLUX_Kkms'] # We want to compare the SUM of the components to our data.
        tauok=numpy.all([dat['TAU']<taumax,dat['TAU']>taumin,dat2['TAU']<taumax,dat2['TAU']>taumin],axis=0)
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
    if ndim==4 and nparams==20:   # Record SLED likelihood for 1 component.
        for s, nd in enumerate(newdat[0:13]): cube[ndim+3+s]=nd
        # cube[ndim+3:]=newdat[0:13] doesn't work
    if ndim==8 and nparams==30: # Record SLED for likelihood for 2 component
        #cube[ndim+9:]=newdat[0:13]
        for s, nd in enumerate(newdat[0:13]): cube[ndim+9+s]=nd
        
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

# A test
testcube=[0.5 for i in range(n_params)]
if n_dims==8: testcube[6]=0.3
myprior(testcube,n_dims,n_params)

# Maybe do a loop over various densities and column densities.
print testcube
print myloglike(testcube,n_dims,n_params)
print testcube
print myloglike_v3(testcube,n_dims,n_params)

