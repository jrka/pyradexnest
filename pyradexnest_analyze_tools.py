# JRK 8/12/13: maketables now determines the mode with the greatest
# total likelihood and outputs 3 additional columns: its mean, sigma, and max.
# JRK 8/14/13: maketables and plotmarginal2 now use meas['addmass'] to 
# convert from log(BACD/dv) to log(mass/dv), meas['addmass'] is now part of the 
# measdata pickle.
# JRK 8/22/13: Fixed plotconditional2 and plotsledto work properly for 1 component modeling.
# JRK 10/30/13: In plot_marginal, allow overplotting of a different model for comparison,
# parameterized by dists2 and cube2.

import numpy as np
import astropy.units as units
from multiplot import multipanel
import matplotlib.pyplot as plt
import pyradex as pyradex
import matplotlib.mlab as mlab

def define_plotting(n_dims,n_sec,sled_to_j,lw,compare=False):
    # Based on the number of parameters, return the information needed for plotting.
    # If this is a comparison, set compare=True, as this changes the colors to be used.

    if n_dims==4:  # 1 component
        parameters = ["h2den1","tkin1","cdmol1","ff1","lum1","press1","bacd1"] 
        add=[0,0,lw,0, lw,0,lw,]
        mult=[1,1,1,1, 1,1,1] # Multiplication is NOT something used for CO RADEX modeling.
        colors=['b','b','b','b','b','b','b']
        if compare: colors2=['k','k','k','k','k','k','k']
        plotinds=[[0,1,2,3],[4,5,6]]
        sledcolors=['b']
        if compare: sledcolors=['#6495ED']
        # Add if we also have likelihoods CO fluxes, append the required information for the likelihood plot.
        for x in range(sled_to_j): 
            parameters.append('flux'+str(x+1))
            add.append(0)
            mult.append(1)
            colors.append('#6495ED') if compare else colors.append('k')
        plotinds.append(range(n_dims+np.sum(n_sec),n_dims+np.sum(n_sec)+sled_to_j,1))
    elif n_dims==8:  # 2 component
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

def get_dists(distfile,s,datsep,grid_points=40):
    import os
    import cPickle as pickle
    n_params=len(s['modes'][0]['mean'])
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
        print datsep.keys()
        for key in datsep.keys():
            print 'Marginalizing...',key
            data=datsep[key]
            tdists={}
            maxes=[]
            for i in range(n_params):
                x,y,z = bin_results(s,data,i,dim2=None,grid_points=grid_points,
                                    marginalization_type='sum', only_interpolate=False)
                z=z.reshape(grid_points)
                tdists[i]=np.vstack((x,z))
                
                # If we are going to do normalization of multiple modes, we need to 
                # keep the "max" of each one easily accessible here.
                maxes.append(z.max())
              
                for j in range(i):
                    x,y,z = bin_results(s,data,i,dim2=j,grid_points=grid_points,
                                        marginalization_type='sum', only_interpolate=False)
                    tdists[i,j]=np.dstack((x,y,z))
            tdists['max']=maxes
            dists[key]=tdists
        
        pickle.dump(dists, open(distfile, "wb") )
        print 'Saved distributions.pkl'
        
    return dists
    
def maketables(s,n_params,parameters,cube,add,mult,title='results',addmass=0,n_dims=8):
    from astropy.table import Table, Column
    
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
        if n_dims==8: 
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

def plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 xr=[[2,6.5],[0.5,3.5],[13,23],[-3,0]],title='',norm1=True,
                 modecolors=[[0,0],[0.1,0.9],[0.2,0.5],[0.3,0.7,1]],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=0,simplify=False,meas=0):
    import matplotlib.mlab as mlab
    mp = multipanel(dims=(np.mod(n_dims,4)/2+2,2),figID=1,padding=(0,0.2),panel_size=(1,np.mod(n_dims,4)/2+1)) # 
    if not simplify: mp.title(title, xy=(0.5, 0.97))
    plt.subplots_adjust(bottom=0.08,left=0.09,right=0.98,top=0.95)
    
    gridinds=[2,3,0,1]
    if n_dims==8: gridinds=[2,3,0,1,2,3,0,1]  # multiplot is now different indices...
    
    if dists2:
        gridinds=[2,3,0,1]
        if n_dims2==8: gridinds=[2,3,0,1,2,3,0,1]  # multiplot is now different indices...
    
    for g,j in enumerate(plotinds[0]):      
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
                mp.grid[gridinds[g]].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m 

                mp.grid[gridinds[g]].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                mp.grid[gridinds[g]].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                mp.grid[gridinds[g]].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                ##print mode['maximum'][j]+add[j],mode['sigma'][j]
            
        yplot=dists['all'][j][1,:] 
        mp.grid[gridinds[g]].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        #mp.grid[gridinds[g]].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
      
    # Comparison distributions overplotted.  
    if dists2:
        for g,j in enumerate(plotinds2[0]):
            yplot2=dists2['all'][j][1,:]
            mp.grid[gridinds2[g]].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
            mp.grid[gridinds2[g]].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')            
    
    # Ranges and labels
    mp.fix_ticks()
    mp.grid[0].set_ylabel("Likelihood")
    mp.grid[2].set_ylabel("Likelihood")
    for i in range(4):
        mp.grid[gridinds[i]].set_xlabel(nicenames[i])  # x-axis labels
        mp.grid[gridinds[i]].set_xlim(xr[i])           # x-axis ranges.
        if norm1: mp.grid[gridinds[g]].set_ylim(0,1)   # y-axis ranges
        
    mp.grid[0].text(4,1,'ln(like):')
    for m,mode in enumerate(s['modes']): 
        try:
            mp.grid[0].text(4,0.9-0.1*m,'%.2f' % mode['local evidence'],color=[modecolors[m][0],modecolors[m][1],1])
        except:
            mp.grid[0].text(4,0.9-0.1*m,'%.2f' % mode[u'local log-evidence'],color=[modecolors[m][0],modecolors[m][1],1])

    mp.draw()
    plt.savefig('fig_marginalized.png')
    print 'Saved fig_marginalized.png'
    
def plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 xr=[[2,6.5],[0.5,3.5],[13,23],[-3,0]],title='',norm1=True,
                 modecolors=['g','m','y','c'],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],meas=0,
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=8,simplify=False):
    mp=multipanel(dims=(1,3),figID=2,padding=(0,0),panel_size=(3,1)) # figID=2,
    plt.clf()
    mp=multipanel(dims=(1,3),figID=2,padding=(0,0),panel_size=(3,1)) # 
    plt.subplots_adjust(bottom=0.12,left=0.06,right=0.98,top=0.88)
    
    if not simplify: mp.title(title, xy=(0.5, 0.97))
    
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
                
                mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
                mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],    
                       color=modecolors[m],label='Mode',linestyle='--')    
                mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],
                       color=modecolors[m],label='Mode',linestyle='--')
            
        yplot=dists['all'][j][1,:]
        mp.grid[np.mod(j-n_dims,3)].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        mp.grid[np.mod(j-n_dims,3)].axvline(x=cube[j]*mult[j]+add[j],color=colors[j])
        if norm1: mp.grid[np.mod(j-n_dims,3)].set_ylim(0,1)   
        
        if j==n_dims:
            ax2a=mp.grid[np.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]*mult[j]+add[j]-np.log10(units.solLum.to('erg/s'))
            ax2a.plot(newx,yplot,'-',color=colors[j], drawstyle='steps')
        
        if j==n_dims+2:
            ax2b=mp.grid[np.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]*mult[j]+add[j]+meas['addmass']
            ax2b.plot(newx,yplot,'-', color=colors[j], drawstyle='steps')
            
    if dists2:
        for j in plotinds2[1]:
            yplot2=dists2['all'][j][1,:]

            mp.grid[np.mod(j-n_dims2,3)].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
            mp.grid[np.mod(j-n_dims2,3)].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison') 
    
    # Ranges and labels
    mp.fix_ticks()
    mp.grid[0].set_ylabel("Relative Likelihood")   
    [mp.grid[i-min([4,5,6]+np.mod(n_dims,4))].set_xlabel(nicenames[i]) for i in [4,5,6]+np.mod(n_dims,4)] 
    mp.grid[1].set_xlim([xr[0][0]+xr[1][0],xr[0][1]+xr[1][1]]) # x-axis ranges.
    mp.grid[2].set_xlim([xr[2][0]+xr[3][0],xr[2][1]+xr[3][1]])
    
    # Add a secondary x-axis for BACD --> Mass
    # log(mass) = log(BACD) + log(area) + log(mu)+log(m_H2) - log(X)
    #     meas['areacm'] + meas['head']['mol_weight'] + log10(2.0*units.M_p.to('kg')/units.solMass.to('kg')) - meas['head']['abundance']
    # Makes sure aligned
    ax2a.set_xlim(mp.grid[0].set_xlim()-np.log10(units.solLum.to('erg/s')))
    ax2a.set_xlabel(r'Log Luminosity  [L$_{\odot}$]')
    ax2b.set_xlim(mp.grid[2].set_xlim()+meas['addmass'])
    ax2b.set_xlabel(r'Log Molecular Mass [M$_{\odot}$]')

    mp.draw() 
    plt.savefig('fig_marginalized2.png')
    print 'Saved fig_marginalized2.png'
    
    if n_dims > 7: # JRK 1/23/14, instead of n_dims == 8
        mp=multipanel(dims=(1,3),figID=3,padding=(0,0),panel_size=(3,1)) #
        if not simplify: mp.title(title,xy=(0.5,0.97))
    
        for j in plotinds[2]:
            for m,mode in enumerate(s['modes']):
                yplot=dists[m][j][1,:]
                #mp.grid[np.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= np.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =np.ravel(zip(yplot, yplot))
                mp.grid[np.mod(j-n_dims,3)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j],color=modecolors[m],label='Mode')
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]+mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
                #mp.grid[np.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]-mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
            
            yplot=dists['all'][j][1,:]
            mp.grid[np.mod(j-n_dims,3)].plot(dists['all'][j][0,:]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
            mp.grid[np.mod(j-n_dims,3)].axvline(x=cube[j]+add[j],color=colors[j])
       
        if dists2:
            for j in plotinds2[2]:
                yplot2=dists2['all'][j][1,:]
                mp.grid[np.mod(j-n_dims2,3)].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                mp.grid[np.mod(j-n_dims2,3)].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')     

        mp.grid[0].set_ylabel("Relative Likelihood")       
        [mp.grid[i-min([7,8,9]+np.mod(n_dims,4))].set_xlabel(nicenames[i]) for i in [7,8,9]+np.mod(n_dims,4)]

        mp.fix_ticks()
        plt.savefig('fig_marginalized2ratio.png')
        print 'Saved fig_marginalized2ratio.png'  
    
def plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 modecolors=['g','m','y','c']):
    import matplotlib.cm as cm
    mp = multipanel(dims=(n_dims-1,n_dims-1),figID=4,diagonal='lower',panel_size=(3,3)) # 

    for i in plotinds[0]:
        for j in range(i):
            #ind=(n_dims-1) * (i-1) + j 
            ind=(n_dims-1)-i+j+(n_dims-2)*(n_dims-1-i) # Wow.
            mp.grid[ind].contourf(dists['all'][i,j][:,:,1]*mult[j]+add[j], dists['all'][i,j][:,:,0]*mult[i]+add[i], dists['all'][i,j][:,:,2], 
                    5, cmap=cm.gray_r, alpha = 0.8,origin='lower') # NEed to transpose?????
                    
            #mp.grid[ind].scatter(datsep['all'][:,j+2]*mult[j]+add[j],datsep['all'][:,i+2]*mult[i]+add[i], color='k',marker='+',s=1, alpha=0.4)
                
            if i==n_dims-1:
                mp.grid[ind].set_xlabel(parameters[j])    
            if j==0:
                mp.grid[ind].set_ylabel(parameters[i])
            
            for m,mode in enumerate(s['modes']):
                mp.grid[ind].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
                mp.grid[ind].axhline(y=mode['mean'][i]*mult[i]+add[i],color=modecolors[m])
            
            mp.grid[ind].axvline(x=cube[j]*mult[j]+add[j],color='k',linestyle='--',label='4D Max')
            mp.grid[ind].axhline(y=cube[i]*mult[i]+add[i],color='k',linestyle='--')

    ###### Still need to get everything aligned right.  Fix ticks moves the outer boxes
    mp.fix_ticks()
    for i in plotinds[0]:
        for j in range(i):
            ind=(n_dims-1)-i+j+(n_dims-2)*(n_dims-1-i) # Wow.
            mp.grid[ind].set_xlim([np.min(dists['all'][i,j][:,:,1]*mult[j]+add[j]),np.max(dists['all'][i,j][:,:,1]*mult[j]+add[j])])
            mp.grid[ind].set_ylim([np.min(dists['all'][i,j][:,:,0]*mult[i]+add[i]),np.max(dists['all'][i,j][:,:,0]*mult[i]+add[i])])
    
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()

    plt.savefig('fig_conditional.png')
    print 'Saved fig_conditional.png'

def plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 modecolors=['g','m','y','c']):
    import matplotlib.cm as cm
    nplot=n_sec[0]-1
    mp = multipanel(dims=(nplot,nplot),figID=5,diagonal=True,panel_size=(3,3)) # 

    for i in plotinds[1]:
        for j in range(n_dims,i):
            #ind=(nplot) * (i-n_dims-1) + j-n_dims
            ind=(nplot)-(i-n_dims)+(j-n_dims)+(nplot-1)*(nplot-(i-n_dims))
            mp.grid[ind].contourf(dists['all'][i,j][:,:,1]+add[j], dists['all'][i,j][:,:,0]+add[i], dists['all'][i,j][:,:,2], 
                    5, cmap=cm.gray_r, alpha = 0.8,origin='lower') # NEed to transpose?????
                
            if i==plotinds[1][nplot]:
                mp.grid[ind].set_xlabel(parameters[j])    
            if j==plotinds[1][0]:
                mp.grid[ind].set_ylabel(parameters[i])
 
            for m,mode in enumerate(s['modes']):
                mp.grid[ind].axvline(x=mode['mean'][j]+add[j],color=modecolors[m],label='Mode')
                mp.grid[ind].axhline(y=mode['mean'][i]+add[i],color=modecolors[m])
            
            mp.grid[ind].axvline(x=cube[j]+add[j],color='k',linestyle='--',label='4D Max')
            mp.grid[ind].axhline(y=cube[i]+add[i],color='k',linestyle='--')
           
    ###### Still need to get everything aligned right.
    mp.fix_ticks()
    for i in plotinds[1]:
        for j in range(n_dims,i):
            ind=(nplot)-(i-n_dims)+(j-n_dims)+(nplot-1)*(nplot-(i-n_dims))
            mp.grid[ind].set_xlim([np.min(dists['all'][i,j][:,:,1]+add[j]),np.max(dists['all'][i,j][:,:,1]+add[j])])
            mp.grid[ind].set_ylim([np.min(dists['all'][i,j][:,:,0]+add[i]),np.max(dists['all'][i,j][:,:,0]+add[i])])
            
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()

    plt.savefig('fig_conditional2.png')
    print 'Saved fig_conditional2.png'
    
def plotsled(meas,cube,n_params,n_dims,modemed,modemax,modesig,plotinds,title='',lum=False,meas2=0,cube2=[],setyr=[0,0],
             colors=['b','r'],colors2=['#6495ED','m'],n_dims2=0,simplify=False,asymerror=0,
             sled_to_j=0):
    # 12/3/13: Fixed bug if meas2 and meas have different number of elements (esp. diff # of non-upper-limits).
    # 4/2/14: Use asymerror=Table if you want to use the assymetric error regions for plotting.
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
   
    # First, plot the data.
    ok=[meas['flux']!=0]
    ul=[meas['flux']==0]

    xplot=meas['J_up']
    yplot=meas['flux']
    yerr=meas['sigma']
    yplot0=yplot
    yerr0=yerr
    
    # Did we do optical depth corrections?  If so, plot: the data with the MEAN lambda0 and beta results AND data with the input 
    #  lambda0 and beta, AND of course the original data.
    if np.mod(n_dims,4)==2:
        label=['Data (Median Corr)','Data (Input Corr)','Data (No Corr)']
        xplot=[xplot,xplot,xplot]
        corr1=tau_corr(meas['J_up'],modemed[n_dims-2],modemed[n_dims-1])
        corr2=tau_corr(meas['J_up'],meas['head']['beta'],meas['head']['lambda0'])
        yplot=[yplot/corr1,yplot/corr2,yplot]
        yerr=[yerr/corr1,yerr/corr2,yerr]
    else: 
        label=['Data']
        xplot=[xplot]
        yplot=[yplot]
        yerr=[yerr]
    
    # And if we have the SLED likelihoods... those are in plotinds[3].  ONLY for the run we are plotting... no comparison.
    if sled_to_j:
        sledinds=plotinds[3] if n_dims==8 else plotinds[2]

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
            import astropy.coordinates.distances as dist
            (yfill1,trash)=spire_conversions(yfill1,'kkms','lsol',range(1,14,1)*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc)
            (yfill2,trash)=spire_conversions(yfill2,'kkms','lsol',range(1,14,1)*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc)       
            yfill1*=meas['head']['lw']
            yfill2*=meas['head']['lw']
        
        if n_dims==4:    
            plt.fill_between(range(1,sled_to_j+1,1), yfill1, yfill2, where=yfill2>=yfill1, facecolor='gray', interpolate=True)
        else:
            plt.fill_between(range(1,sled_to_j+1,1), yfill1[0:sled_to_j], yfill2[0:sled_to_j], 
                where=yfill2[0:sled_to_j]>=yfill1[0:sled_to_j], facecolor='b', interpolate=True,alpha=0.2)
            plt.fill_between(range(1,sled_to_j+1,1), yfill1[sled_to_j:], yfill2[sled_to_j:], 
                where=yfill2[sled_to_j:]>=yfill1[sled_to_j:], facecolor='r', interpolate=True,alpha=0.2)

    if lum:
        from pysurvey import spire_conversions
        import astropy.coordinates.distances as dist
        (yplot,yerr)=spire_conversions(yplot,'kkms','lsol',xplot*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc,inerr=yerr)
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
        ok2=[meas2['flux']!=0]
        ul2=[meas2['flux']==0]

        xplot2=meas2['J_up']
        yplot2=meas2['flux']
        yerr2=meas2['sigma']
        if lum:
            (yplot2,yerr2)=spire_conversions(yplot2,'kkms','lsol',xplot2*115.3,sr=meas2['head']['omega_s'],
                              z=meas2['head']['z'],dist=dist.Distance(z=meas2['head']['z']).Mpc,inerr=yerr2)
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
    chisq=[]
    t_n_params=n_params # Temporary n_params, incase we have to replace it with 2
    t_n_dims=n_dims
    t_colors=colors
    for w in w_in:
        if w==0: 
            thiscube=cube
            thismeas=meas
            thisok=ok
            linestyle='-'
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
        
        dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=2.73, species=thismeas['head']['mol'], velocity_gradient=1.0, debug=False)
        dat['flux_kkms']*=np.power(10,thiscube[3])
        model1=dat['flux_kkms']
        
        if np.mod(n_dims,4)==2: model1/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
        if lum: 
            (model1,trash)=spire_conversions(model1,'kkms','lsol',dat['J_up']*115.3,sr=thismeas['head']['omega_s'],
                              z=thismeas['head']['z'],dist=dist.Distance(z=thismeas['head']['z']).Mpc)
            model1*=thismeas['head']['lw']
        plt.plot(dat['j_up'],model1,color=t_colors[0],label=label1,linestyle=linestyle)
        newdat=dat['flux_kkms']
    
        if t_n_dims>7:
            dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
                           temperature=np.power(10,thiscube[5]), column=np.power(10,thiscube[6]), 
                           collider_densities={'H2':np.power(10,thiscube[4])},
                           tbg=2.73, species=thismeas['head']['mol'], velocity_gradient=1.0, debug=False)
            dat2['flux_kkms']*=np.power(10,thiscube[7])
            model2=dat2['flux_kkms']
            if np.mod(n_dims,4)==2: model2/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
            if lum:
                (model2,trash)=spire_conversions(model2,'kkms','lsol',dat2['J_up']*115.3,sr=thismeas['head']['omega_s'],
                              z=thismeas['head']['z'],dist=dist.Distance(z=thismeas['head']['z']).Mpc)
                model2*=thismeas['head']['lw']  

            model3=model1+model2
            newdat=dat['flux_kkms']+dat2['flux_kkms']
            plt.plot(dat2['j_up'],model2,color=t_colors[1],label=label2,linestyle=linestyle)
            plt.plot(dat2['j_up'],model3,color='k',label=labelT,linestyle=linestyle)           


        # Calculate Chi Squared?
        # Need to match "newdat" with meas['flux']
        if np.mod(n_dims,4)==2: newdat/=tau_corr(dat['J_up'],cube[n_dims-2],cube[n_dims-1])
        chisq.append(0)
        for i,tflux in enumerate(thismeas['flux'][thisok]):
            temp=newdat[dat['j_up'] == thismeas['J_up'][thisok][i]]
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
    if meas2:
        plt.ylim([np.min(np.concatenate((yplot0[yplot0>0]-1.0*yerr0[yplot0>0],yplot2[yplot2>0]-1.0*yerr2[yplot2>0]),axis=0)),
              np.max(np.concatenate((yplot0[yplot0>0]+1.0*yerr0[yplot0>0],yplot2[yplot2>0]+1.0*yerr2[yplot2>0]),axis=0))])
    else: 
       plt.ylim([np.min(yplot0[yplot0>0]-1.0*yerr0[yplot0>0]),
              np.max(yplot0[yplot0>0]+1.0*yerr0[yplot0>0])])
    # Overrride that ylim if setyr is used.
    if np.sum(setyr) != 0: plt.ylim(setyr)
    yr=np.log10(plt.ylim())
    if not simplify: # only print Chi Sq if simplify is not True.
        for i,c in enumerate(chisq): 
            pos=np.power(10,(0.2-0.05*i)*(yr[1]-yr[0])+yr[0])
            plt.text(2,pos,c)
    plt.legend()
    
    plt.savefig('fig_sled.png')
    print 'Saved fig_sled.png'
    
    ######## OPTICAL DEPTH
    plt.figure(7)
    plt.clf()
    plt.xlabel('Upper J')
    plt.ylabel('Tau')
    if not simplify: plt.title(title)

    # Ugh, dat is whatever was last used, must call RADEX again...
    thiscube=cube
    thismeas=meas
    
    dat=pyradex.pyradex(minfreq=1, maxfreq=1600,
                          temperature=np.power(10,cube[1]), column=np.power(10,cube[2]), 
                          collider_densities={'H2':np.power(10,cube[0])},
                          tbg=2.73, species=thismeas['head']['mol'], velocity_gradient=1.0, debug=False)    
    
    if n_dims==8: dat2=pyradex.pyradex(minfreq=1, maxfreq=1600,
                           temperature=np.power(10,thiscube[5]), column=np.power(10,thiscube[6]), 
                           collider_densities={'H2':np.power(10,thiscube[4])},
                           tbg=2.73, species=thismeas['head']['mol'], velocity_gradient=1.0, debug=False)
    
    plt.plot(dat['j_up'],dat['tau'],color=colors[0],label='Component 1')
    if n_dims==8: plt.plot(dat2['j_up'],dat2['tau'],color=colors[1],label='Component 2')

    plt.xlim([0,14])
    plt.legend()

    plt.savefig('fig_tau.png')
    print 'Saved fig_tau.png'
    
    ######## EXCITATION TEMPERATURE
    plt.figure(8)
    plt.clf()
    plt.xlabel('Upper J')
    plt.ylabel('Excitation Temperature [K]')
    if not simplify: plt.title(title)

    plt.plot(dat['j_up'],dat['t_ex'],color=colors[0],label='Component 1')
    plt.axhline(y=np.power(10,cube[1]),color=colors[0],linestyle='--')
    if n_dims==8: 
        plt.plot(dat2['j_up'],dat2['t_ex'],color=colors[1],label='Component 2')
        plt.axhline(y=np.power(10,cube[5]),color=colors[1],linestyle='--')

    plt.xlim([0,14])
    #plt.ylim([0,1e3])
    plt.legend()

    plt.savefig('fig_tex.png')
    print 'Saved fig_tex.png'
    
def plotmarginalsled(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title='',norm1=True,
                 modecolors=[[0,0],[0.1,0.9],[0.2,0.5],[0.3,0.7,1]],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[],
                 plotinds2=0,add2=0,mult2=0,n_dims2=0):

    if n_dims > 7: # Two component, use 7 to account for beta/tau sometimes in 1 component.
        useind=3
    else: 
        useind=2
    
    sled_to_j=len(plotinds[useind])
    if n_dims==8: sled_to_j=sled_to_j/2
    dims=np.array((np.ceil(np.sqrt(sled_to_j)),np.ceil(sled_to_j/np.ceil(np.sqrt(sled_to_j)))))
    mp = multipanel(dims=dims.astype(int),figID=9,padding=(0,0.2),panel_size=dims.astype(int)) # 
    mp.title(title,xy=(0.5,0.97))
    
    gridinds=np.mod(range(sled_to_j),dims[0].astype(int)) # wow, seriously...
    count=dims[1]-1
    for i in range(dims[1].astype(int)):
        print i,count,row
        gridinds[i*dims[0]:dims[0]+i*dims[0]]+=count*dims[0]
        count-=1
    if n_dims==8: gridinds=np.concatenate([gridinds,gridinds])
    
    
    for j in plotinds[useind]:      
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
                mp.grid[gridinds[j]].fill_between(xx-dx,0,yy,color=modecol,alpha=0.2) # color=modecolors[m

                mp.grid[gridinds[j]].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecol,label='Mode')
                mp.grid[gridinds[j]].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                mp.grid[gridinds[j]].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecol,label='Mode',linestyle='--')
                ##print mode['maximum'][j]+add[j],mode['sigma'][j]
           
        yplot=dists['all'][j][1,:] 
        mp.grid[gridinds[j]].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        mp.grid[gridinds[j]].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
        
        if norm1: mp.grid[gridinds[j]].set_ylim(0,1)
        
    if dists2:
        if n_dims2 > 7: # Two component, use 7 to account for beta/tau sometimes in 1 component.
            useind2=3
        else: 
            useind2=2
        try: 
            for j in plotinds2[useind2]:
                yplot2=dists2['all'][j][1,:]
                mp.grid[gridinds[j]].plot(dists2['all'][j][0,:]*mult2[j]+add2[j], yplot2, '-', color=colors2[j], drawstyle='steps')
                mp.grid[gridinds[j]].axvline(x=cube2[j]*mult2[j]+add2[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')          
        except:
            print 'No SLED distribution comparison to overplot'  
    
    mp.global_ylabel("Relative Likelihood")
    
    [mp.grid[i].set_xlabel(nicenames[i+4+3+3+np.mod(n_dims,4)]) for i in range(len(plotinds[useind]))]

    mp.fix_ticks()
    plt.savefig('fig_marginalizedsled.png')
    print 'Saved fig_marginalizedsled.png'

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