import os, threading, subprocess
import numpy as np
#from astropy.table import Table,join
import pyradex as pyradex
import cPickle as pickle
import matplotlib.pyplot as plt

# JRK 9/29/15: modify measdata_pickle to allow a 9th line in the header
# to explicitly set the luminosity distance.

# read in measdata ######################################################################
def measdata_pickle(measdatafile,sled_to_j=False,taulimit=[-0.9,100.0],tbg=2.73):
    from astropy import constants as const
    from astropy import units as units
    import astropy.coordinates.distances as dist
    import astropy.io.ascii as ascii
    
    if not taulimit: taulimit=[-0.9,100.0]
    
    jytocgs=1.0e-23
    top=np.genfromtxt(measdatafile,delimiter="\n",comments="#")
    head={'omega_s':top[0],'z':top[1],'mag':top[2],'lw':top[3],'abundance':top[4],
        'dynmass':top[5],'mol_weight':top[6],'length':top[7]}
    # If set, add in the luminosity distance in Mpc. Else, calculate it.
    if np.isfinite(top[8]):
        head['dl']=top[8]
        data_start=9
    else: 
        head['dl']=dist.Distance(z=head['z']).Mpc
        data_start=8
    
    data=ascii.read(measdatafile,delimiter='\s',comment='#',guess=False,Reader=ascii.NoHeader,
       data_start=data_start,names=['mol','J_up','J_lo','freq','FLUX_Jykms','measerr','calerr'])
    
    # Add the molecule name to the header.  ONE MOLECULE FOR THIS.  JRK 1/28/15
    if len(set(data['mol'])) != 1:
        raise Exception('This routine only set to run ONE molecule at a time; check your measdata.txt file.')
    else:
        head['mol']=data['mol'][0].lower()
    
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
    
    # Require at least 2 unique lines to model.
    if len(np.unique(jup)) < 2: raise Exception("Not enough lines to model.")
           
    # Also calculate the "prior cuts" for mass and length (in "cube" units), add that to the meas structure.
    # Based off of rtl3_multi_make_prior_cuts.pro
    # Convert luminosity distance in Mpc from header to pc.
    angdist=(head['dl']*1e6)*(1.0+head['z'])**2 # angdist=dist.Distance(z=head['z']).pc/(1.0+head['z'])**2
    s_area=head['omega_s']*(angdist)**2/(1.0+head['z'])**2 # pc^2
    if head['dynmass'] != -1:
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
            'sigmacut':3.0, 'taulimit':taulimit, 'addmass':addmass, 'sled_to_j':sled_to_j, 'tbg':tbg}
    print meas
    
    pickle.dump(meas, open("measdata.pkl", "wb") )

# prior function ########################################################################
# The prior function is in the config file.  This allows you to choose different ranges
# for parameters with different runs.

# loglikelihood function ################################################################
def myloglike(cube, ndim, nparams):
    import warnings
    
    debug=False # Change this to return likelihoods to diagnose certain problems
    
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
               if debug: return 1e2
               return -2e100
    else:
        if cube[2]+cube[3] > meas['masscut'] or cube[2]-cube[0]-0.5*cube[3] > meas['lengthcut']:
            if debug: return 1e2
            return -2e100

    # Call RADEX for the first component.
    try:
        dat1=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=meas['tbg'], species=meas['head']['mol'], velocity_gradient=1.0, debug=False,
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
        #     tbackground=meas['tbg'],species=meas['head']['mol'],deltav=1.0,debug=False)
        #niter=R.run_radex(validate_colliders=False)
        #model1=1.064575*R.T_B*np.power(10,cube[3]) # Integrating over velocity, and Filling Factor
        #model1=model1.value # Hatred of units in my way
        #tau1=R.tau
        #jup1=R.upperlevelindex        
        #cube[ndim]=np.sum(R.source_brightness_beta) # luminosity; not correct units yet.
        
    except:
        if debug: return 1e3
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
                               tbg=meas['tbg'], species=meas['head']['mol'], velocity_gradient=1.0, debug=False,
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
            if debug: return 1e3
            return -2e100
    else:
        modelt=model1# total model
        tauok=np.all([tau1<taumax,tau1>taumin],axis=0)
    
    # Check that we have at least one line with optical depth in allowable limits.
    if not np.any(tauok[jmeas]):
        if debug: return 1e4
        return -2e100
        
    # Check that we don't violate ANY line flux upper limits.
    ul=np.where(meas['flux']==0)[0]
    for i in ul:
        ulok=modelt[jup1==meas['J_up'][i]] < meas['sigma'][i]*sigmacut
        if not ulok: 
            if debug: return 1e5
            return -2e100
    
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
    # The public distribution of RADEX will not output this luminosity; sorry all users...
    if 'LogLmol' in dat1.keys():
        cube[ndim]=dat1['LogLmol']+meas['areacm']+cube[2]+cube[3]
    else: 
        cube[ndim]=1.0  # Those not using our private RADEX code will not get luminosity.
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
        if 'LogLmol' in dat2.keys():
            cube[ndim+3]=dat2['LogLmol']+meas['areacm']+cube[6]+cube[7]
        else: 
            cube[ndim+3]=1.0  # See luminosity note above.
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
    
# check mass and length limits ###########################################################
def plot_mass_length_limits(meas,ndims,pmin,pmax,measdatafile=''):
    from mpl_toolkits.mplot3d import Axes3D

    # meas['lengthcut'] and meas['masscut'] tell us the cuts in terms of our parameters, 
    # but not in physical units.
    if measdatafile!='':  
        top=np.genfromtxt(measdatafile,delimiter="\n",comments="#")

    # Cold component mass limit. Technically it is the sum of the 2 we worry about.
    fig=plt.clf()
    plt.fill_between([pmax[2],meas['masscut']-pmax[3]],[meas['masscut']-pmax[2],pmax[3]],pmax[3],
        label='Restricted by Mass Limit')
    plt.xlim(pmin[2],pmax[2])
    plt.ylim(pmin[3],pmax[3])
    plt.xlabel('Component 1 Column Density per Unit Linewidth')
    plt.ylabel('Component 1 Filling Factor')
    if measdatafile!='':
        plt.title('{:.2e}'.format(top[5])+' Msolar Mass Cut: Restricts High (N+ff), Shaded')
    else: 
        plt.title('Mass Cut: Restricts High (N+ff), Shaded')
    plt.savefig('prior_mass.png')


    # Length limit for warm and cold
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(pmax[2],pmin[2])
    ax.set_ylim(pmin[0],pmax[0])
    ax.set_zlim(pmin[3],pmax[3])
    ax.set_xlabel('Column Density per Unit Linewidth')
    ax.set_ylabel('Collider Density')
    ax.set_zlabel('Filling Factor')
    
    xarr=np.array([ [pmin[2],pmin[2]], [pmax[2],pmax[2]]])
    yarr=np.array([ [pmin[0],pmax[0]], [pmin[0],pmax[0]]])
    zarr=(meas['lengthcut']-(xarr-yarr))/(-0.5)
    if zarr[0,0]<pmin[3]: xarr[0,0]=meas['lengthcut']+yarr[0,0]+0.5*pmin[3]
    if zarr[1,1]<pmin[3]: yarr[1,1]=xarr[1,1]-0.5*pmin[3]-meas['lengthcut']
    if zarr[0,1]<pmin[3]: yarr[0,1]=np.nan
    zarr=np.clip(zarr,pmin[3],pmax[3])
    ax.plot_wireframe(xarr,yarr,zarr)
    if measdatafile!='':
        ax.set_title('{:.0f}'.format(top[7])+' pc Length Cut: Restricts High N, Low n, Low ff')
    else: 
        ax.set_title('Length Cut: Restricts High N, Low n, Low ff')
    plt.savefig('prior_length.png')

    # ALL parameter space is restricted
    #if pmin[2]+pmin[3]<meas['masscut']: 



