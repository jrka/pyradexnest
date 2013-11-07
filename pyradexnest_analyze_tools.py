# JRK 8/12/13: maketables now determines the mode with the greatest
# total likelihood and outputs 3 additional columns: its mean, sigma, and max.
# JRK 8/14/13: maketables and plotmarginal2 now use meas['addmass'] to 
# convert from log(BACD/dv) to log(mass/dv), meas['addmass'] is now part of the 
# measdata pickle.
# JRK 8/22/13: Fixed plotconditional2 and plotsledto work properly for 1 component modeling.
# JRK 10/30/13: In plot_marginal, allow overplotting of a different model for comparison,
# parameterized by dists2 and cube2.

import numpy
import astropy.units as units
from multiplot import multipanel
import matplotlib.pyplot as plt
import pyradexv3 as pyradex

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
    
    postdat=numpy.loadtxt(post_file)
    breakind.insert(0,0)
    breakind.append(len(postdat))
    nmodes=len(breakind)-1
    datsep={}
    for i in range(nmodes):datsep[i]=postdat[breakind[i]:breakind[i+1]]
    
    return datsep

def arrowloglen(ylim,top,frac=32.0):
    yr=numpy.log10(ylim)
    delta=(yr[1]-yr[0])/frac
    bottom=10.0**(numpy.log10(top)-delta)
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
    
    grid_x = numpy.mgrid[min1:max1:n*1j]
    if dim2 is not None:
        m = n
        grid_x, grid_y = numpy.mgrid[min1:max1:n*1j, min2:max2:n*1j]
        binsize2 = (max2 - min2) / m
    else:
        m = 1
        grid_x = numpy.mgrid[min1:max1:n*1j]
        grid_y = [0]
        
    binsize1 = (max1 - min1) / n
    
    dim1_column = data[:,2 + dim1]
    if dim2 is not None:
        dim2_column = data[:,2 + dim2]
        coords = numpy.array([dim1_column, dim2_column]).transpose()
    else:
        coords = dim1_column.transpose()
    values = data[:,0]
    if use_log_values:
        values = numpy.log(values)
    grid_z = numpy.zeros((n,m))
    minvalue = values.min()
    maxvalue = values.max()
    
    # for each grid item, find the matching points and put them in.
    for row, col in itertools.product(range(len(grid_x)), range(len(grid_y))):
        if dim2 is not None:
            xc = grid_x[row,col]
            here_x = numpy.abs(dim1_column - xc) < binsize1 / 2.
            yc = grid_y[row,col]
            here_y = numpy.abs(dim2_column - yc) < binsize2 / 2.
        else:
            xc = grid_x[row]
            here_x = numpy.abs(dim1_column - xc) < binsize1 / 2.
            here_y = True
        
        bin = values[numpy.logical_and(here_x, here_y)]
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
    if os.path.exists(distfile) and os.path.getmtime(distfile) > os.path.getmtime('chains/1-.json'):
        dists=pickle.load(open(distfile,"rb"))
    else:
        # We want 1D distributions for each parameter, 2D for each combination of parameters
       # The secondary paramters (luminosity, pressure, bacd) are included here.
        dists={}
        print datsep.keys()
        for key in datsep.keys():
            print 'Marginalizing...',key
            data=datsep[key]
            tdists={}
            for i in range(n_params):
                x,y,z = bin_results(s,data,i,dim2=None,grid_points=grid_points,
                                    marginalization_type='sum', only_interpolate=False)
                z=z.reshape(grid_points)
                tdists[i]=numpy.vstack((x,z))
                for j in range(i):
                    x,y,z = bin_results(s,data,i,dim2=j,grid_points=grid_points,
                                        marginalization_type='sum', only_interpolate=False)
                    tdists[i,j]=numpy.dstack((x,y,z))
            dists[key]=tdists
        
        pickle.dump(dists, open(distfile, "wb") )
        print 'Saved distributions.pkl'
        
    return dists
    
def maketables(s,n_params,parameters,cube,add,mult,title='results',addmass=0,n_dims=8):
    from astropy.table import Table, Column
    
    # If there are multiple modes, determine the most likely one.
    localev=[]
    for j in s['modes']: localev.append(j['local evidence'])
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

    # But for latex, just need 2 digits.
    table['Median'].format='%.2f' # I am aware this is stupid.
    table['-1 Sigma'].format='%.2f'
    table['+1 Sigma'].format='%.2f'
    table['Best Fit'].format='%.2f'
    table['Mode Mean'].format='%.2f'
    table['Mode Sigma'].format='%.2f'
    table['Mode Maximum'].format='%.2f'
    
    table.pprint()
    table.write("results_latex.tex",format="latex",caption=title.replace('_',' '))
    
    return table
###############################################################################################

def plotmarginal(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title='',norm1=True,
                 modecolors=['g','m','y','c'],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],
                 dists2={},colors2=['g','g','g','g','m','m','m','m','g','g','g','m','m','m','gray','gray','gray'],cube2=[]):
    mp = multipanel(dims=(2,2),figID=1,padding=(0,0.2))
    mp.title(title)
    
    for j in plotinds[0]:
        # The filling was not adding up right; don't plot each individual mode.  FUTURE NOTE: normalization is messed up
        # because numpy arrays are pointers, and I've modified the value of dists. 
        for m,mode in enumerate(s['modes']):
        #    yplot=dists[m][j][1,:]
        #    yplot*=numpy.exp(mode['local evidence'])/numpy.exp(s['global evidence'])
        #    if norm1: yplot/=numpy.max(dists['all'][j][1,:])
        #    #mp.grid[numpy.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
        #    dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
        #    xx= numpy.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
        #    yy =numpy.ravel(zip(yplot, yplot))
        #    #mp.grid[numpy.mod(j,4)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
        #    
            mp.grid[numpy.mod(j,4)].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
            mp.grid[numpy.mod(j,4)].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
            mp.grid[numpy.mod(j,4)].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
            ##print mode['maximum'][j]+add[j],mode['sigma'][j]
            
        yplot=dists['all'][j][1,:]   # WHY DOES DISTS CHANGE SOMEWHERE HERE?
        if norm1: yplot/=numpy.max(dists['all'][j][1,:])
        mp.grid[numpy.mod(j,4)].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        mp.grid[numpy.mod(j,4)].axvline(x=cube[j]*mult[j]+add[j],color=colors[j],linestyle='-',label='4D Max')
        
        if dists2:
            yplot2=dists2['all'][j][1,:]
            if norm1: yplot2/=numpy.max(dists2['all'][j][1,:])
            mp.grid[numpy.mod(j,4)].plot(dists2['all'][j][0,:]*mult[j]+add[j], yplot2, '-', color=colors2[j], drawstyle='steps')
            mp.grid[numpy.mod(j,4)].axvline(x=cube2[j]*mult[j]+add[j],color=colors2[j],linestyle='-',label='4D Max, Comparison')            
        
        if norm1: mp.grid[numpy.mod(j,4)].set_ylim(0,1)
    
    mp.grid[0].set_ylabel("Probability")
    mp.grid[2].set_ylabel("Probability")
    
    if parameters[0]=='h2den1': # CO likelihood
        mp.grid[0].set_xticks([2,3,4,5,6])
        mp.grid[0].set_xticks([2.5,3.5,4.5,5.5],minor=True)
        mp.grid[1].set_xticks([1,1.5,2,2.5,3.0])
        mp.grid[3].set_xticks([-2.0,-1.0])
        mp.grid[3].set_xticks([-2.5,-1.5,-0.5],minor=True)
        #[mp.grid[i].set_xlim(xrange[i]) for i in range(4)]
        [mp.grid[i].set_xlabel(nicenames[i]) for i in range(4)]
    else: 
        [mp.grid[i].set_xlabel(nicenames[i]) for i in plotinds[0]]
    mp.grid[0].text(4,1,'ln(like):')
    for m,mode in enumerate(s['modes']): mp.grid[0].text(4,0.9-0.1*m,'%.2f' % mode['local evidence'],color=modecolors[m])

    mp.fix_ticks()
    plt.savefig('fig_marginalized.png')
    print 'Saved fig_marginalized.png'
    
def plotmarginal2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,title='',norm1=True,
                 modecolors=['g','m','y','c'],colors=['b','b','b','b','r','r','r','r','b','b','b','r','r','r','k','k','k'],meas=0):
    mp=multipanel(dims=(1,3),figID=2,padding=(0,0),panel_size=(3,1))
    mp.title(title)
    
    for j in plotinds[1]:        
        # The filling was not adding up right; don't plot each individual mode.
        for m,mode in enumerate(s['modes']):
        #    yplot=dists[m][j][1,:]
        #    yplot*=numpy.exp(mode['local evidence'])/numpy.exp(s['global evidence'])
        #    if norm1: yplot/=numpy.max(dists['all'][j][1,:])
        #    #mp.grid[numpy.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
        #    dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
        #    xx= numpy.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
        #    yy =numpy.ravel(zip(yplot, yplot))
        #    mp.grid[numpy.mod(j-n_dims,3)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
            
            mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j],color=modecolors[m],label='Mode')
            mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]+mode['sigma'][j],
                   color=modecolors[m],label='Mode',linestyle='--')
            mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]*mult[j]+add[j]-mode['sigma'][j],
                   color=modecolors[m],label='Mode',linestyle='--')
            
        yplot=dists['all'][j][1,:]
        if norm1: yplot/=numpy.max(yplot)
        mp.grid[numpy.mod(j-n_dims,3)].plot(dists['all'][j][0,:]*mult[j]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
        mp.grid[numpy.mod(j-n_dims,3)].axvline(x=cube[j]*mult[j]+add[j],color=colors[j])
        if norm1: mp.grid[numpy.mod(j-n_dims,3)].set_ylim(0,1)
        
        if j==n_dims:
            ax2a=mp.grid[numpy.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]-numpy.log10(units.solLum.to('erg/s'))
            ax2a.plot(newx,yplot,'-',color=colors[j], drawstyle='steps')
        
        if j==n_dims+2:
            ax2b=mp.grid[numpy.mod(j-n_dims,3)].twiny()
            newx=dists['all'][j][0,:]+add[j]+meas['addmass']
            ax2b.plot(newx,yplot,'-', color=colors[j], drawstyle='steps')
 
    # Add a secondary x-axis for BACD --> Mass
    # log(mass) = log(BACD) + log(area) + log(mu)+log(m_H2) - log(X)
    #     meas['areacm'] + meas['head']['mol_weight'] + log10(2.0*units.M_p.to('kg')/units.solMass.to('kg')) - meas['head']['abundance']
	# Makes sure aligned
    ax2a.set_xlim(mp.grid[0].set_xlim()-numpy.log10(units.solLum.to('erg/s')))
    ax2a.set_xlabel(r'log(L [L$_{sol}$])')
    ax2b.set_xlim(mp.grid[2].set_xlim()+meas['addmass'])
    ax2b.set_xlabel(r'log(M(H$_{2}$ [M$_{sol}$])')

    mp.grid[0].set_ylabel("Probability")       
    [mp.grid[i-4].set_xlabel(nicenames[i]) for i in [4,5,6]]

    mp.fix_ticks()
    plt.savefig('fig_marginalized2.png')
    print 'Saved fig_marginalized2.png'
    
    if n_dims==8:
        mp=multipanel(dims=(1,3),figID=3,padding=(0,0),panel_size=(3,1))
        mp.title(title)
    
        for j in plotinds[2]:
            for m,mode in enumerate(s['modes']):
                yplot=dists[m][j][1,:]
                yplot*=numpy.exp(mode['local evidence'])/numpy.exp(s['global evidence'])
                if norm1: yplot/=numpy.max(dists['all'][j][1,:])
                #mp.grid[numpy.mod(j,4)].plot(dists[m][j][0,:]*mult[j]+add[j],yplot,':',color=modecolors[m],label='Mode',drawstyle='steps')
                dx=(dists[m][j][0,1]-dists[m][j][0,0])*mult[j]
                xx= numpy.ravel(zip(dists[m][j][0,:]*mult[j]+add[j], dists[m][j][0,:]*mult[j]+add[j] + dx))
                yy =numpy.ravel(zip(yplot, yplot))
                mp.grid[numpy.mod(j-n_dims,3)].fill_between(xx-dx,0,yy,color=modecolors[m],alpha=0.2)
                #mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j],color=modecolors[m],label='Mode')
                #mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]+mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
                #mp.grid[numpy.mod(j-n_dims,3)].axvline(x=mode['mean'][j]+add[j]-mode['sigma'][j],color=modecolors[m],label='Mode',linestyle='--')
            
            yplot=dists['all'][j][1,:]
            if norm1: yplot/=numpy.max(yplot)
            mp.grid[numpy.mod(j-n_dims,3)].plot(dists['all'][j][0,:]+add[j], yplot, '-', color=colors[j], drawstyle='steps')
            mp.grid[numpy.mod(j-n_dims,3)].axvline(x=cube[j]+add[j],color=colors[j])

        mp.grid[0].set_ylabel("Probability")       
        [mp.grid[i-7].set_xlabel(nicenames[i]) for i in [7,8,9]]

        mp.fix_ticks()
        plt.savefig('fig_marginalized2ratio.png')
        print 'Saved fig_marginalized2ratio.png'    

def plotconditional(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 modecolors=['g','m','y','c']):
    import matplotlib.cm as cm
    mp = multipanel(dims=(n_dims-1,n_dims-1),diagonal=True,figID=4,panel_size=(3,3))

    for i in plotinds[0]:
        for j in range(i):
            ind=(n_dims-1) * (i-1) + j
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
            ind=(n_dims-1) * (i-1) + j
            mp.grid[ind].set_xlim([numpy.min(dists['all'][i,j][:,:,1]*mult[j]+add[j]),numpy.max(dists['all'][i,j][:,:,1]*mult[j]+add[j])])
            mp.grid[ind].set_ylim([numpy.min(dists['all'][i,j][:,:,0]*mult[i]+add[i]),numpy.max(dists['all'][i,j][:,:,0]*mult[i]+add[i])])
    
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()

    plt.savefig('fig_conditional.png')
    print 'Saved fig_conditional.png'

def plotconditional2(s,dists,add,mult,parameters,cube,plotinds,n_sec,n_dims,nicenames,
                 modecolors=['g','m','y','c']):
    import matplotlib.cm as cm
    if n_dims > 7:
        nplot=n_sec[0]-1
    else:
        nplot=n_sec-1
    mp = multipanel(dims=(nplot,nplot),diagonal=True,figID=5,panel_size=(3,3))

    for i in plotinds[1]:
        for j in range(n_dims,i):
            ind=(nplot) * (i-n_dims-1) + j-n_dims
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
            ind=(nplot) * (i-n_dims-1) + j-n_dims
            mp.grid[ind].set_xlim([numpy.min(dists['all'][i,j][:,:,1]+add[j]),numpy.max(dists['all'][i,j][:,:,1]+add[j])])
            mp.grid[ind].set_ylim([numpy.min(dists['all'][i,j][:,:,0]+add[i]),numpy.max(dists['all'][i,j][:,:,0]+add[i])])
            
    plt.legend(bbox_to_anchor=(0.,1.02,1.,0.102),loc=3)
    plt.draw()

    plt.savefig('fig_conditional2.png')
    print 'Saved fig_conditional2.png'

def plotsled(meas,cube,n_params,modemed,modemax,title='',lum=False):
    plt.figure(6)
    plt.clf()
    plt.xlabel('Upper J')
    if lum:
        plt.ylabel(r'Luminosity [L$_\odot$]')
    else: 
        plt.ylabel('K (per km/s)')
    plt.yscale('log')
    plt.title(title)
   
    # First, plot the data.
    ok=[meas['flux']!=0]
    ul=[meas['flux']==0]

    xplot=meas['J_up']
    yplot=meas['flux']
    yerr=meas['sigma']
    if lum:
        from pysurvey import spire_conversions
        import astropy.coordinates.distances as dist
        (yplot,yerr)=spire_conversions(yplot,'kkms','lsol',xplot*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc,inerr=yerr)
        yplot*=meas['head']['lw']
        yerr*=meas['head']['lw']
    
    plt.errorbar(xplot[ok],yplot[ok],yerr=yerr[ok],
                 color='black',fmt='^',linestyle='None',label='Data')
    
    arrlen =arrowloglen(plt.ylim(),3.0*yerr[ul]       ,frac=32.0)
    headlen=arrowloglen(plt.ylim(),3.0*yerr[ul]+arrlen,frac=64.0)
    
    for i, l in enumerate(arrlen):
        plt.arrow(xplot[ul][i], 3.0*yerr[ul][i], 0.0*xplot[ul][i], l, 
                  fc='black', ec='black', head_width=0.3, head_length=-headlen[i],overhang=1) 
    
    # Next, plot 4DMax, then the mode, then Mode Maximum.
    chisq=[]
    for w in [0,1,2]:
        if w==0: 
            thiscube=cube
            linestyle='-'
            label1='Component 1'
            label2='Component 2'
            label3='Total (Best Fit)'
        if w==1: 
            thiscube=modemax
            linestyle='--'
            label1=''
            label2=''
            label3='Total (Mode Maximum)'
        if w==2: 
            thiscube=modemed
            linestyle=':'
            label1=''
            label2=''
            label3='Total (Mode Median)'
        dat=pyradex.pyradex(flow=1, fhigh=1600,
                      tkin=numpy.power(10,thiscube[1]), column_density=numpy.power(10,thiscube[2]), 
                      collider_densities={'H2':numpy.power(10,thiscube[0])},
                      tbg=2.73, molecule='co', velocity_gradient=1.0, debug=False)
        dat['FLUX_Kkms']*=numpy.power(10,thiscube[3])
        model1=dat['FLUX_Kkms']
        if lum: 
            (model1,trash)=spire_conversions(model1,'kkms','lsol',dat['J_up']*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc)
            model1*=meas['head']['lw']
        plt.plot(dat['J_up'],model1,color='b',label=label1,linestyle=linestyle)
        newdat=dat['FLUX_Kkms']
    
        if n_params>7:
            dat2=pyradex.pyradex(flow=1, fhigh=1600,
                           tkin=numpy.power(10,thiscube[5]), column_density=numpy.power(10,thiscube[6]), 
                           collider_densities={'H2':numpy.power(10,thiscube[4])},
                           tbg=2.73, molecule='co', velocity_gradient=1.0, debug=False)
            dat2['FLUX_Kkms']*=numpy.power(10,thiscube[7])
            model2=dat2['FLUX_Kkms']
            if lum:
                (model2,trash)=spire_conversions(model2,'kkms','lsol',dat2['J_up']*115.3,sr=meas['head']['omega_s'],
                              z=meas['head']['z'],dist=dist.Distance(z=meas['head']['z']).Mpc)
                model2*=meas['head']['lw']  
           
            model3=model1+model2
            newdat=dat['FLUX_Kkms']+dat2['FLUX_Kkms']
            
            
            plt.plot(dat2['J_up'],model2,color='r',label=label2,linestyle=linestyle)
            plt.plot(dat2['J_up'],model3,color='gray',label=label3,linestyle=linestyle)

        # Calculate Chi Squared?
        # Need to match "newdat" with meas['flux']
        chisq.append(0)
        for i,tflux in enumerate(meas['flux'][ok]):
             temp=newdat[dat['J_up'] == meas['J_up'][ok][i]]
             #print meas['J_up'][ok][i],tflux,meas['sigma'][ok][i],temp
             chisq[w]+=((temp-tflux)/meas['sigma'][ok][i])**2.0
      
    # A bit more tweaking of the plot:
    plt.xlim([0,14])
    #plt.ylim([1e-3,1e1])
    plt.ylim([numpy.min(yplot[yplot>0]-1.0*yerr[yplot>0]),
              numpy.max(yplot[yplot>0]+1.0*yerr[yplot>0])])
    yr=numpy.log10(plt.ylim())
    for i,c in enumerate(chisq): 
        pos=numpy.power(10,(0.2-0.05*i)*(yr[1]-yr[0])+yr[0])
        plt.text(2,pos,c)
    plt.legend()
    
    plt.savefig('fig_sled.png')
    print 'Saved fig_sled.png'
    
    plt.figure(7)
    plt.clf()
    plt.xlabel('Upper J')
    plt.ylabel('Tau')
    plt.title(title)

    plt.plot(dat['J_up'],dat['TAU'],color='b',label='Component 1')
    if n_params>7: plt.plot(dat2['J_up'],dat2['TAU'],color='r',label='Component 2')

    plt.xlim([0,14])
    plt.legend()

    plt.savefig('fig_tau.png')
    print 'Saved fig_tau.png'
