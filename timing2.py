import pyradex
import numpy as np

@profile
def func1(n):
    R = pyradex.Radex(collider_densities={'oH2':900,'pH2':100}, column=n,debug=True)
    R.run_radex()
    T = R.tex
    
    T2=pyradex.pyradex(collider_densities={'oH2':900,'pH2':100}, column=n, debug=True)
    
    table=R(collider_densities={'oH2':900,'pH2':100}, column=n)

for n in 10**np.arange(12,18): func1(n)

#python -m cProfile -o output.pstats timing2.py
#import pstats
#p = pstats.Stats('output.pstats')
#p.strip_dirs() #otherwise it's way too long to read.
#p.sort_stats('cumulative') # see other choices http://docs.python.org/2/library/profile.html#module-pstats
#p.print_stats(0.01) # only the top 1%

#p.sort_stats('tottime') # see other choices http://docs.python.org/2/library/profile.html#module-pstats
#p.print_stats(0.01) # only the top 1%
