# Configuration file for PyRadexNest
#
#
# 1) Number of dimensions.  4 = one component model, 8 = two component model.
n_dims=4
#
# 2) You can also calculate distributions for CO fluxes of any lines up to this J number.
#        Or just say false.
sled_to_j=5
#
# 3) Define the parameter ranges.  Here you are converting a unit cube to real parameter
#    ranges, which are all in log space.
#    cube[i]=cube[i]*(max-min)+min
def myprior(cube, ndim, nparams):
    cube[0]=cube[0]*4.5+2  # h2den1  2 to 6.5
    # Temperature 1 range will depend on if this is 1 component (full temp range) or 2 component (limited temp range)
    cube[2]=cube[2]*7+12   # cdmol1  12 to 19
    cube[3]=cube[3]*3-3    # ff1     -3 to 0
    if ndim>4:
        cube[1]=cube[1]*2.2+0.5# tkin1   0.5 to 2.3 = 3.16 to 502 K.
        cube[4]=cube[4]*4.5+2  # h2den2  2 to 6.5
        cube[5]=cube[5]*1.5+2  # tkin2   2 to 3.5 = 100 to 3162 K
        cube[6]=cube[6]*7+12   # cdmol2  12 to 19
        cube[7]=cube[7]*3-3    # ff2     -3 to 0
    else:
        cube[1]=cube[1]*3+0.5
# 4) Normalize the plots so that the maximum of a marginalized plot is 1?
norm1=True
# 5) Lower and upper limits for tau; lines outside this range will not be used.
taulimit=[-0.9,100]
# 6) Background temperature to use in RADEX? If not specified, default of 2.73 K is used.
tbg=2.73