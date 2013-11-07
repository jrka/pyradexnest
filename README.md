pyradexnest
===========

Use Pymultinest, Multinest, and Pyradex to fit spectral energy line distributions and plot the results.

===========
You need:

1) Multinest
Newest version:
http://ccpforge.cse.rl.ac.uk/gf/project/multinest/
Version compatible with current version of PyMultiNest:
git clone https://github.com/JohannesBuchner/MultiNest.git MultiNest

2) PyMultiNest
git clone git://github.com/JohannesBuchner/PyMultiNest

3) pyradex (github, Adam Ginsburg)

4) multiplot (github, Jordan Mirocha)

And the aforementioned python packages, plus this folder,
 must be in your python path.
 
 ** Make sure tolerance for PyMultiNest is set such as -2e100 is not utilized in likelihood **
 
UPDATES THAT NEED TO BE MADE TO PYRADEX.PYRADEX  BEFORE THIS WILL WORK:
1) Return just a dictionary, not an astropy table, e.g. 

	data_in_floats = [map(float,col) for col in data_in_columns] #  We can get a ValueError if not floats (ie E+101)
	data={key: np.array(value) for (key,value) in zip(header_names,data_in_floats)}

2) Add to that dictionary the total luminosity.  Is this output by command line radex?

	data['LogLmol']=float(Lpermol[0].split('=')[1])
 
===========

To run:

1) Enter the directory of your run.  (e.g. example_run)

2) Create a file which contains the data for fitting.  Its name needs to be measdata.txt, in the example_run folder. Format of this file:
-8 lines of header information:
-- Beam or source area in steradians
-- Redshift
-- Magnification
-- Linewidth in km/s
-- CO/H2 relative abundance
-- Maximum dynamical mass (solar masses)
-- mu
-- Maximum length (pc)
- A table with 7 columns:
-- Molecule (must all be the same for now, and correspond to the Molecule.dat file) And in fact, this only works for CO for now.
-- J_upper
-- J_lower
-- Observed frequency (GHz)
-- Integrated flux in Jy km/s
-- Measured error in flux in Jy km/s
-- Percent calibration error to add in quadrature (set to 0 if not desired).
            
3) Setup a few things in pyradexnest.py 
-- Number of dimensions, n_dims: 4 for 1 component modeling, 8 for 2 component.
-- If desired, check the limits for the parameters as defined in the mycube function.
    
4) RUN IT, use mpi if desired.  BE IN YOUR DIRECTORY.  mpirun -np 8 python PATH/TO/pyradexnest.py
    
5) Analyze it.  BE IN YOUR DIRECTORY. python PATH/TO/pyradexnest_analyze.py
        
