pyradexnest
===========

Use Pymultinest, Multinest, and Pyradex to fit spectral energy line distributions 
and plot the results.

New, May 2016: pymultinest\_multimol.py (and its buddy, pyradexnest\_tools\_multimol.py) 
can be used for multiple species or isotopologue modeling (e.g. simultaneously modeling 12CO, 13CO, and C18O). 
The code assumes these arise from the same gas conditions, with one additional free parameter per species, its relative number abundance to the primary species.
The primary species is the one listed FIRST in the measdata.txt table.
The analysis scripts should be compatible with output from both pyradexnest.py and pyradexnest\_multimol.py.
Furthermore, pymultinest\_multimol.py should be compatible with single species modeling, if n_mol=1 is set.

===========
## You need:

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


And the aforementioned python packages, plus *this* folder,
 must be in your python path.  You also need astropy and matplotlib.
 
 ** Make sure tolerance for PyMultiNest is set such as -2e100 is not utilized in likelihood **
 ** Luminosity not yet implemented **

===========
## To run:

1) Enter the directory of your run (e.g. example_run). Note that the remaining operation 
assumes '.' in your python path, as measdata.txt and config.py (and future files) need to be found.

2) Create a file which contains the data for fitting.  Its name needs to be measdata.txt, in the example_run folder. This file is the same whether using single or multiple molecule modeling.
Format of this file:

- 8 (or 9) lines of header information:
    - Beam or source area in steradians
    - Redshift
    - Magnification
    - Linewidth in km/s
    - CO/H2 relative abundance
    - Maximum dynamical mass (solar masses)
    - mu
    - Maximum length (pc)
    - Optional line: luminosity distance in Mpc. If not set, this will be calculated from the redshift
    using astropy.coordinates.distances.Distance. Useful for very low redshift.
- A table with 7 columns:
    - Molecule (must correspond to molecule.dat filename [case insensitive?] used by pyradex)
    - J_upper
    - J_lower
    - Observed frequency (GHz)
    - Integrated flux in Jy km/s
    - Measured error in flux in Jy km/s
    - Percent calibration error to add in quadrature (set to 0 if not desired).
            
3) Create a config.py file also in that directory, with the following:
- Number of dimensions, n_dims: 4 for 1 component modeling, 8 for 2 component.
- If you'd like to include likelihood distributions for the line fluxes themselves, 
   set sled_to_j as a nonzero number: you will calculate distributions for all lines up
to and including this J_upper value.
- Define the "myprior" function, see example.
- If you'd like to normalize the marginalized distributions so peak =1, norm=True
- taulimit, an upper and lower limit for acceptable optical depths. Lines outside of this 
range will not be used for likelihood calculations. The RADEX manual does not recommend 
optical depths above 100, where the assumptions for the escape probability method break down.
- tbg, the dust temperature to use in RADEX. Default is 2.73 K if not specified.
- If using pymultinest_multimol.py, you will need to add some additional things:
    -  n_mol, number of molecules/isotopologues being used. 
    - Make sure the "myprior" function includes the correct range for each molecule's relative abundance.
    
4) RUN IT, use mpi if desired.  BE IN YOUR DIRECTORY.  

```
mpirun -np 4 python PATH/TO/pyradexnest.py
```

To run the example, for instance:
```
cd example_run
mpirun -np 4 python ../pyradexnest.py
```
    
Replace pyradexnest.py with pyradexnest\_multimol.py for multi-molecule modeling.
    
5) Analyze it. While still in your same working directory, run pyradexnest_analyze.py
```
cd example_run
python ../pyradexnest_analyze.py
```

This creates:

- distributions.pkl (Not newly created each time.)
- Multiple .png figures:
    - fig_conditional.png
    - fig_conditional2.png
    - fig_marginalized.png
    - fig_marginalized2.png
    - fig_marginalized2ratio.png (if multiple components)
    - fig_margalinzedsled.png (if sled_to_j nonzero)
    - fig_sled.png
    - fig_tau.png
    - fig_tex.png
- Two results tables:
    - results_ascii.txt
    - results_latex.tex

You can combine these files using the template summary_indv.tex.


===========
## FAQ... preliminary

1) How long does it take?  

This varies.  I recently ran the following (Mac OS X v10.9.5, 3.2 GHz Intel Core i5, 32 GB RAM):
1 component, 1 molecule,  without MPI, 10 minutes
2 components, 1 molecule, mpirun -np 4, 1 hour

If you've left a simulation running while you 
were away, you can check how long it took by comparing the modification time of measdata.pkl 
(written at the start of the program) and that of most things in the chains folder, e.g. 1-stats.dat, 
which is last modified when it finished.

2) How is the redshift used?

The redshift is used in the conversion from Jy km/s to K, though this will make very little difference 
at low redshifts.  The most important use of the redshift is to determine an angular size scale 
to convert the area from sr to physical area (e.g. pc^2), which is necessary for calculating the mass.
Now, however, you can explicitly set the luminosity distance to use instead.

3) Which molecules can I use?

Any molecule that RADEX can handle where transitions are uniquely described by J_upper.  