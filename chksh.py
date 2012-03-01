#!/usr/bin/env python
import sys
import os
import scipy.special
import scipy.linalg
import math
import pylab as pl
import numpy as np
from scipy.lib.lapack import get_lapack_funcs
#import dpbtf2

beta   = 4. # in km/s
rho    = 3. # gm/cm^3 (need to account for units in source)
r      = np.arange(0.,6372.,1.)
l      = 75
m      = 1
i_src  = 6321
r_src  = r[i_src]
print "r_src=%g" % r_src
mu     = rho*beta*beta
mu_src = rho*beta*beta 
r_srf  = r[-1]
mrt    = 1.
mrp    = 1.
mtt    = 1.
mpp    = 1.
mtp    = 1.
Qm     = 1000.
tau    = 1024.
omega  = complex(2.*math.pi/1000.,0.)#1./tau)

xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
beta   = beta*xdsm
xdsm   = xdsm*xdsm
mu     = mu*xdsm
mu_src = mu_src*xdsm

jn = np.zeros(len(r))
yn = np.zeros(len(r))
for i in range(0,len(r)):
    tmp = scipy.special.sph_jn(l,omega*r[i]/beta)
    # Store first bessel function
    jn[i] = tmp[0][l]
    if (i == i_src): 
        j_src  = tmp[0][l]
        jp_src = tmp[1][l]*omega/beta
    elif i == len(r)-1: 
        j_srf  = tmp[0][l]
        jp_srf = tmp[1][l]*omega/beta
    if (i >= i_src):
        tmp = scipy.special.sph_yn(l,omega*r[i]/beta)
        # Store second bessel function
        yn[i] = tmp[0][l]
    if (i == i_src): 
        y_src  = tmp[0][l]
        yp_src = tmp[1][l]*omega/beta
    elif i == len(r)-1: 
        y_srf  = tmp[0][l]
        yp_srf = tmp[1][l]*omega/beta



b1 = math.sqrt((2*l+1)/(16.*math.pi))
b2 = math.sqrt((2*l+1)*(l-1)*(l+2)/(64.*math.pi))
# Set up jumps in W and T
dW = 0.+0.j
if (abs(m) == 1):
    dW = b1*complex(m*mrp,mrt)/(r_src*r_src*mu_src)
    print "dW = (%g,%g)" % (dW.real,dW.imag)
dTw = 0.+0.j
if (abs(m) == 2):
    dTw = b2*complex(-2*mtp,math.copysign(mpp-mtt,m))/r_src
    print "dTw = (%g,%g)" % (dTw.real,dTw.imag)

# Linear system for boundary conditions at source depth adn surface
#              below source             above source
A = np.array([[ -j_src+0j ,       j_src+0j,             y_src+0j],\
              [ -jp_src+0j ,      jp_src+0j,            yp_src+0j],\
              [0.+0j,        jp_srf-j_srf/r_srf+0j, yp_srf-y_srf/r_srf+0j]])
# Set boundary condition terms - why is there and r_src**2 here?
b = np.zeros((3,1),complex)
b[0] = dW
b[1] = dTw/(mu_src*r_src*r_src)
b[2] = 0.+0j
# Solve for the coefficients
x = np.linalg.solve(A,b)
# soltuion coefficeitns in x[0-3]


#for i in range(0,3):
#    print b[i],A[i][0]*x[0]+A[i][1]*x[1]+A[i][2]*x[2]

# W0 is the analytical solution
W0 = np.zeros(len(r),complex)
# below source
W0[0:i_src] = x[0]*jn[0:i_src]
# Above source
W0[i_src:] = x[1]*jn[i_src:] + x[2]*yn[i_src:]


# Print some informatoin on analytical solution
print "W = ",x[0]*jn[i_src],x[1]*jn[i_src]+x[2]*yn[i_src]
print "Tsrf=",mu*(W0[len(r)-1]-W0[len(r)-2] - W0[len(r)-1]/r_srf)
print "delta(W) = ",W0[i_src]-W0[i_src-1]
print "delta(Tw) = ",mu_src*(W0[i_src+1]-2.*W0[i_src]+W0[i_src-1])
#,W0[i_src+1]-W0[i_src],x[1]*jp_src+x[2]*yp_src,W0[i_src]-W0[i_src-1],x[0]*jp_src
#print jn[i_src]-jn[i_src-1], jp_src,yn[i_src+1]-yn[i_src], yp_src,

########            Now for the DSM Solution
######## Function to calculate fundamental element matrices on rg[]
######## Not that ll matrices have the diagonal [1] index, off-diagnal in [0] index
def Imats(rg):

    n   = len(rg)
    j   = 0
    if rg[0] == 0.: 
        n = len(rg)-1
        j = -1
    I0  = np.zeros((2,n))
    I1  = np.zeros((2,n))
    I2a = np.zeros((2,n))
    I2b = np.zeros((2,n))
    I3  = np.zeros((2,n))

    for i in range(0,len(rg)-1):
        r_i0 = rg[i]
        r_i1 = rg[i+1]
        dr   = r_i1 - r_i0
# I0: r^2*X*X
        if r_i0 != 0.:
            I0[1][ j ]  +=  dr*(dr*(dr*0.2 - r_i1*0.5) +r_i1*r_i1/3.)
            I0[0][j+1]   =  dr*(dr*dr*0.05 + r_i0*r_i1/6.)
        I0[1][j+1]   =  dr*(dr*(dr*0.2 + r_i0*0.5) +r_i0*r_i0/3.)
# I1: X*X 
        if r_i0 != 0.:
            I1[1][ j ]  +=  dr/3.
            I1[0][j+1]   =  dr/6.
        I1[1][j+1]   =  dr/3.
# I2a: r*X^dot*X
        if r_i0 != 0.:
            I2a[1][ j ] -=  (r_i1 + 2.*r_i0)/6.
            I2a[0][j+1]  = -(r_i0 + 2.*r_i1)/6.
        I2a[1][j+1]  =  (r_i0 + 2.*r_i1)/6.
# I2b: r*X*X^dot
        if r_i0 != 0.:
            I2b[1][ j ] -=  (r_i1 + 2.*r_i0)/6.
            I2b[0][j+1]  =  (r_i1 + 2.*r_i0)/6.
        I2b[1][j+1]  =  (r_i0 + 2.*r_i1)/6.
# I3: r^2*X^dot*X^dot
        if r_i0 != 0.:
            I3[1][ j ]  +=  (r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
            I3[0][j+1]   = -(r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
        I3[1][j+1]   =  (r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
# Increment column index
        j = j+1
        
    return (I0,I1,I2a,I2b,I3) 
        


dr= 1.
rg = np.arange(0.,6372.,dr)
# Calculate elemental matrices
(I0,I1,I2a,I2b,I3) = Imats(rg)
# Calculate stiffness matrix (straight from paper)
Stiff = -(omega*omega*rho*I0 - mu*((l*(l+1)-1)*I1 -I2a - I2b + I3))

# Calculate Cholesky decomposition
C = scipy.linalg.cholesky_banded(Stiff)


# Find source radius
for i_src in range(len(rg)-1,-1,-1):
    if r_src >= rg[i_src]: break
print 'source radius = %g (%g)' % (rg[i_src],r_src)


# Set up source vector
ncol = Stiff.shape[1]
g = np.zeros((ncol,1),complex)
if abs(m) == 1:
    (J0,J1,J2a,J2b,J3) = Imats(rg[i_src:])
    A = -(omega*omega*rho*J0 - mu*((l*(l+1)-1)*J1 -J2a - J2b + J3))
    # Get source vecotr by multiplyiung A times discontinuous 
    g[i_src-1] = -dW*(A[1][0]+A[0][1])
    for j in range (i_src,ncol-1):
        k = j-i_src+1
        g[j] = -dW*(A[0][k]+A[1][k]+A[0][k+1])
    g[ncol-1] = -dW*(A[1][A.shape[1]-1]+A[0][A.shape[1]-1])
else:
    g[i_src-1] = -dTw
    
# Solve for displacement vector
x = scipy.linalg.cho_solve_banded((C,False),g)
# For dW != 0, add in Dw again to displacement above source 
if abs(m) == 1: x[i_src-1:] += dW

# All done! print out some statistics
print "W=",x[i_src-2],x[i_src-1]
print len(rg)
print "Tsrf=",mu*(x[len(rg)-2]-x[len(rg)-3] - x[len(rg)-2]/r_srf)
#for j_src in range(i_src-2,i_src+3):
#    print Stiff[0][j_src-1]*x[j_src-2] + Stiff[1][j_src-1]*x[j_src-1]+Stiff[0][j_src]*x[j_src]
print "delta(W) =",x[i_src-1]-x[i_src-2]
print "delta(Tw) = ",mu_src*(x[i_src]-2.*x[i_src-1]+x[i_src-2])
#pl.plot(rg[0:-1],real(x))
#######
#pl.figure(1)
#pl.plot()
#pl.figure(2)
# Plot analytical vs. numerical solutions
pl.plot(r,W0.real,r,W0.imag+1.e-9,rg[1:],x.real,rg[1:],x.imag+1.e-9)
pl.show()

