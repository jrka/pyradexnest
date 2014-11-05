# Test the python radex version timings

import timeit
import numpy as np
import textwrap

import pyradex
import pyradexv3
import numpy as np


%timeit R=pyradex.Radex(collider_densities={'oH2':900,'pH2':100}, column=1e16, species='co',temperature=40)
%timeit R.run_radex()
%timeit T=R.tex

# this is in fact longer in total than the previous...
%timeit Tlvg=R(collider_densities={'oH2':900,'pH2':100}, column=1e16, species='co',temperature=40)
%timeit T2=pyradex.pyradex(collider_densities={'oH2':900,'pH2':100}, column=1e16, species='co',temperature=40)
%timeit dat=pyradexv3.radex(flow=1, fhigh=1600, tkin=40, column_density=numpy.power(10,16.0), collider_densities={'H2':numpy.power(10,3.0)}, tbg=2.73, molecule='co', velocity_gradient=1.0, debug=False)

for n in 10**np.arange(12,18):
    setup = "import pyradex"
    ptiming = timeit.Timer(stmt="pyradex.pyradex(collider_densities={'oH2':900,'pH2':100},column=%e)" % n,setup=setup).repeat(3,10)
    print "Python: ",np.min(ptiming)
    setup = """
    import pyradex
    R = pyradex.Radex(collider_densities={'oH2':900,'pH2':100}, column=%e)""" % n
    ftiming = timeit.Timer(stmt="R.run_radex(); T = R.tex",setup=textwrap.dedent(setup)).repeat(3,10)
    print "Fortran: ",np.min(ftiming)
    #dominated by array creation...
    ftiming2 = timeit.Timer(stmt="R(collider_densities={'oH2':900,'pH2':100}, column=%e)" % n, setup=textwrap.dedent(setup)).repeat(3,10)
    #print "Fortran (call method): ",np.min(ftiming2)
    print "py/fortran: ",np.min(ptiming)/np.min(ftiming)#,"py/fortran (call method): ",np.min(ptiming)/np.min(ftiming2)
