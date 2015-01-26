pyradexnest
===========

Use Pymultinest, Multinest, and Pyradex to fit spectral energy line distributions and plot the results.

===========
You need:

1) Multinest
Newest version:
http://ccpforge.cse.rl.ac.uk/gf/project/multinest/
Version compatible with current version of PyMultiNest:
```
git clone https://github.com/JohannesBuchner/MultiNest.git MultiNest
```
As of 10/30/14, I have been using MultiNest v3.5.
Currently testing with v3.8 and the most up to date PyMultiNest as of 1/26/15.
You will also want e.g. openmpi available before installing for parallel processing.

2) PyMultiNest
git clone git://github.com/JohannesBuchner/PyMultiNest

New users of this code: make sure you have the previous two components 
working together nicely before proceeding, e.g. you can run 

```
python pymultinest_demo.py 
```
and ideally
```
mpirun -np 2 python pymultinest_demo.py
```

3) pyradex (Adam Ginsburg)
https://github.com/keflavich/pyradex
This will include an installation of radex.

4) multiplot (Jordan Mirocha)
https://bitbucket.org/mirochaj/multiplot


And the aforementioned python packages, plus *this* folder,
 must be in your python path.
 
 ** Make sure tolerance for PyMultiNest is set such as -2e100 is not utilized in likelihood **
 ** Luminosity not yet implemented **

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
            
3) Create a config.py file also in that directory, with the following:
-- Number of dimensions, n_dims: 4 for 1 component modeling, 8 for 2 component.
-- If you'd like to include likelihood distributions for the line fluxes themselves, 
   set sled_to_j as a nonzero number: you will calculate distributions for all lines up
to and including this J_upper value.
-- Define the "myprior" function, see example.
-- If you'd like to normalize the marginalized distributions so peak =1, norm=True
    
4) RUN IT, use mpi if desired.  BE IN YOUR DIRECTORY.  

```
mpirun -np 4 python PATH/TO/pyradexnest.py
```

To run the example, for instance:
```
cd example_run
mpirun -np 2 python ../pyradexnest.py
```
    
5) Analyze it - not done yet.