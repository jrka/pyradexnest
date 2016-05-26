import pymultinest
#from astropy.table import Table,join
from pyradexnest_tools_multimol import *

# JRK 1/12/16: Copied from pyradexnest.py for multiple molecules


def show(filepath): 
	""" open the output (pdf) file for the user """
	if os.name == 'mac': subprocess.call(('open', filepath))
	elif os.name == 'nt': os.startfile(filepath)
	elif os.name == 'posix': subprocess.call(('open', filepath)) # JRK changed 'xdg-open' to 'open'

# Run this in the output directory!  Make "chains" before running with mpi.
if not os.path.exists("chains"): os.mkdir("chains")

######################################################################################
# main function ######################################################################

# User settings loaded into config.py already, otherwise use defaults
# Because taulimit was added later, old config.py files might not have it.
# Default will be used instead in measdata_pickle.
taulimit=None
try:
    from config import *
except:
    print 'Could not load config.py, using default configurations.'
    import shutil
    print os.path.dirname(os.path.abspath(__file__))+'/config_default.py'
    print os.getcwd()+'/config.py'
    shutil.copy2(os.path.dirname(os.path.abspath(__file__))+'/config_default.py',
          os.getcwd()+'/config.py')
    from config import *

try:
    tbg
except:
    print 'tbg not defined in config file; this is a new feature. Using 2.73 K.'
    tbg=2.73

# The file to read in
cwd=os.getcwd()
split=cwd.rpartition('/')
measdatafile='measdata.txt'

# number of dimensions our problem has. Do by n_comp, not n_dims
if n_comp==1:
    parameters=["h2den1","tkin1","cdmol1","ff1"]
    for i in range(n_mol-1): parameters.append('xmol'+str(i+1))
    for i in ["lum1","press1","bacd1"]: parameters.append(i)
elif n_comp==2:
    parameters = ["h2den1","tkin1","cdmol1","ff1","h2den2","tkin2","cdmol2","ff2"]
    for i in range(n_mol-1): parameters.append('xmol'+str(i+1)+'c')
    for i in range(n_mol-1): parameters.append('xmol'+str(i+1)+'w')
    for i in ["lum1","press1","bacd1","lum2","press2","bacd2",
              "lumratio","pressratio","bacdratio"]: parameters.append(i)
else: 
    raise Exception('You must choose n_comp = 1 or 2 components only; check your config.py file.')

# Add SLED likelihood parameters, for EACH molecule. Main molecule = 0
if sled_to_j and n_comp==1: 
    for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m) for i in range(1,sled_to_j+1)])
if sled_to_j and n_comp==2: 
    for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m)+"c" for i in range(1,sled_to_j+1)]) # Cold
    for m in range(n_mol): map(parameters.append,["k"+str(i)+'mol'+str(m)+"w" for i in range(1,sled_to_j+1)]) # Warm
n_params = len(parameters)

# Created our measured data pickle.
measdata_pickle(measdatafile,sled_to_j=sled_to_j,taulimit=taulimit,n_mol=n_mol,tbg=tbg)

# Before starting, record the parameter limits, min and max of prior cube.
# Outdated; now this is recorded in config.py.  But in case you've changed it...
pmin=np.zeros(n_dims)
pmax=np.zeros(n_dims)+1
myprior(pmin,n_dims,n_params)
myprior(pmax,n_dims,n_params)
np.savetxt('prior_cube.txt',[pmin,pmax],fmt='%+.3e')

# Examine the parameter space which is restricted by the length and mass priors.
#meas=pickle.load(open("measdata.pkl","rb"))
#plot_mass_length_limits(meas,n_dims,pmin,pmax,measdatafile=measdatafile)

# Test
#pmid=np.zeros(n_dims)+0.5
#myprior(pmid,n_dims,n_params)
#cube=np.zeros(n_params)
#cube[0:n_dims]=pmid
#print myloglike(cube, n_dims,n_params)

# Still avoiding the progress plotter.
#progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/1-'); progress.start()
#threading.Timer(2, show, ["chains/1-phys_live.points.pdf"]).start() # delayed opening
# run MultiNest
pymultinest.run(myloglike, myprior, n_dims, n_params=n_params, importance_nested_sampling = False, resume = True, 
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
#for i in range(n_params):
#       outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
#       p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
#       plt.ylabel("Probability")
#       plt.xlabel(parameters[i])
#       plt.savefig(outfile, format='pdf', bbox_inches='tight')
#       plt.close()
#       
#       outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
#       p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
#       plt.ylabel("Cumulative probability")
#       plt.xlabel(parameters[i])
#       plt.savefig(outfile, format='pdf', bbox_inches='tight')
#       plt.close()
#
#print "take a look at the pdf files in chains/" 