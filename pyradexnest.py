import pymultinest
#from astropy.table import Table,join
from pyradexnest_tools import *

# JRK 8/13/15: Moved measdata_pickle and myloglike functions to pyradexnest_tools.py.
#   If a config.py file is not found, copy the file config_default.py into the 
#   working directory. This is because the analyze script needs to know the configs later.
# JRK 8/31/15: Commented out 1--mode-marginal-X.pdf and 1--mode-marginal-cumulative-X.pdf
#   plot creation at the end of this script. Prefer not to include, but user may want.
# JRK 9/16/15: User can now set taulimit in the config file; was hardcoded as [-0.9,100.0].
#   These are the limits between which we will trust RADEX and use lines in our likelihood.
#   No need for import statements at top; they are in pyradexnest_tools.py.
# JRK 9/29/15: Added meas['head']['dl'] in Mpc, can be calculated from redshift, or set
#   as 9th header line, see measdata_pickle in pyradexnest_tools.py


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
measdata_pickle(measdatafile,sled_to_j=sled_to_j,taulimit=taulimit,tbg=tbg)

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