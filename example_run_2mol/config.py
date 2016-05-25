# Configuration file for PyRadexNest
#
#
# 1) Number of dimensions.  4 + n_mol - 1 = one component model, 8 + 2*n_mol - 2 = two component model.
#    You cannot use more than 4 molecules.
n_comp=2
n_mol=2
n_dims=4*n_comp+(n_comp)*(n_mol-1)
#
# 2) You can also calculate distributions for CO fluxes of any lines up to this J number.
#        Or just say false.
sled_to_j=13
#
# 3) Define the parameter ranges.  Here you are converting a unit cube to real parameter
#    ranges, which are all in log space.
#    cube[i]=cube[i]*(max-min)+min
def myprior(cube, ndim, nparams):
    # We cannot pass n_comp and n_mol, but can figure those out.
    cube[0]=cube[0]*4.5+2  # h2den1  2 to 6.5
    # Temperature 1 range will depend on if this is 1 component (full temp range) or 2 component (limited temp range)
    cube[2]=cube[2]*7+12   # cdmol1  12 to 19
    cube[3]=cube[3]*3-3    # ff1     -3 to 0
    if ndim>=8:
        n_comp=2
        cube[1]=cube[1]*2.2+0.5# tkin1   0.5 to 2.3 = 3.16 to 502 K.
        cube[4]=cube[4]*4.5+2  # h2den2  2 to 6.5
        cube[5]=cube[5]*1.5+2  # tkin2   2 to 3.5 = 100 to 3162 K
        cube[6]=cube[6]*7+12   # cdmol2  12 to 19
        cube[7]=cube[7]*3-3    # ff2     -3 to 0
        
    else:
        n_comp=1
        cube[1]=cube[1]*3+0.5
    # Add X_mol for additional molecules past the first one.
    # Here all are 1e-4 to 1e-1
    # MORE EXPLANATION HERE.
    n_mol=ndim/n_comp-3
    for i in range(n_mol-1):  # First component
        cube[n_comp*4+i]=cube[n_comp*4+i]*3-4
    if n_comp==2:
        for i in range(n_mol-1): # Second component
            cube[n_comp*4+i+n_mol-1]=cube[n_comp*4+i+n_mol-1]*3-4
# 4) Normalize the plots so that the maximum of a marginalized plot is 1?
norm1=True