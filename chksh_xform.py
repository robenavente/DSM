#!/usr/bin/env python
import sys
import os
import scipy.special
import scipy.linalg
import math
import pylab as pl
import numpy as np
from scipy.lib.lapack import get_lapack_funcs
from scipy.sparse import csc_matrix,linalg
import dsm #,imats_new
import shtools_gssh

#General variables:

mrt    = 1.
mrp    = 1.
mtt    = 1.
mpp    = 1.
mtp    = 1.
Qm     = 1000.
tau    = 1024.

beta   = 4. # in km/s
rho    = 3. # gm/cm^3 (need to account for units in source)
#*print "r_src=%g" % r_src
mu     = rho*beta*beta
mu_src = rho*beta*beta     

Lmax   = 100
PERIOD = np.arange(1000.0,0.,-10.) #Vector with the periods
L      = range(1,Lmax) #Vector with l
phi = 25.
THETA  = np.arange(10.,91.,10.)
thetaphis =zip(THETA,phi*np.ones(len(THETA)))
ps = []
dps = []
for theta,phi in thetaphis:
    z = math.cos(theta*math.pi/180.)
    p,dp = shtools_gssh.plmbar_d1_m2(Lmax, z)
    ps.append(p)
    dps.append(dp)

r_src  = 6321 

#Theoretical solution's variables
rres   = 1.      #Resolution in km  (must agree with the source)
r      = np.arange(0.,6372., rres)
i_srf  = len(r)-1
r_srf  = r[-1]
#Numerical solution's variables
dr     = 1. # Resolution (km) above the source
nb     = 5000 #Gridpoints below the source

	
rbelow = np.linspace(0, r_src-dr, nb)
rabove = np.arange(r_src, r_srf+1,dr)
rg     = np.hstack((rbelow, rabove))
# Redefining i_src to use it in the numerical solution
i_src    = len(rbelow)
for i_src in range(len(rg)-1,-1,-1):
    if r_src >= rg[i_src]: break

w = []
for period in PERIOD:

    w.append([])
    omega  = complex(2.*math.pi/period,1./tau)        
    xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
    #xdsm   = 1.+0j
    beta_dsm   = beta*xdsm
    xdsm   = xdsm*xdsm
    mu_dsm     = mu*xdsm
    mu_src_dsm = mu_src*xdsm

    z = np.zeros((3,5,L+1),complex)
    for l in L:
    
        ERROR  = []   # Resetting the vector with the errors
                
        ########            Now for the DSM Solution
        ######## Function to calculate fundamental element matrices on rg[]
        ######## Not that ll matrices have the diagonal [1] index, off-diagnal in [0] index
        ########


        # Calculate elemental matrices
###########  ###### 

	(K0,K1,K2a,K2b,K3) = dsm.imats(rg,i_src)
	Stiff = -(omega*omega*rho*K0 - mu_dsm*((l*(l+1)-1)*K1 -K2a - K2b + K3))
	ncol = Stiff.shape[1]

	# to csc
	Stiff_data = np.zeros((3*ncol-2),complex)
	Stiff_ij   = np.zeros((2,3*ncol-2))
	j = 0
	for col in range(0,ncol):
	    Stiff_data[j] = Stiff[1][col]
	    Stiff_ij[0,j] = col
	    Stiff_ij[1,j] = col
	    j += 1
	#    print col,j,3*ncol-2
	    if col < ncol-1:
		Stiff_data[j] = Stiff[0][col+1]
		Stiff_ij[0,j] = col
		Stiff_ij[1,j] = col+1
		j += 1
		Stiff_data[j] = Stiff[0][col+1]
		Stiff_ij[0,j] = col+1
		Stiff_ij[1,j] = col
		j += 1
	#print Stiff_data.shape,Stiff_ij.shape
	Stiff_csc = csc_matrix((Stiff_data,Stiff_ij))
	#

	# Superlu
	lu = linalg.splu(Stiff_csc)

	ncol = Stiff.shape[1]
        b1 = math.sqrt((2*l+1)/(16.*math.pi))
        b2 = math.sqrt((2*l+1)*(l-1)*(l+2)/(64.*math.pi))
        for m = range(-2,2,1):
            # Set up jumps in W and T
            dW = 0.+0.j
            if (abs(m) == 1):
                dW = b1*complex(m*mrp,mrt)/(r_src*r_src*mu_src_dsm)
            #*print "dW = (%g,%g)" % (dW.real,dW.imag)
            dTw = 0.+0.j
            if (abs(m) == 2):
                dTw = b2*complex(-2*mtp,math.copysign(mpp-mtt,m))/r_src
            #*print "dTw = (%g,%g)" % (dTw.real,dTw.imag)
            g = np.zeros((ncol),complex)
            A = Stiff[:,i_src:]
            if abs(m) == 1:
                g[i_src:-1] =  -dW*(A[0][:-1]+A[1][:-1]+A[0][1:])
                g[-1] = -dW*(A[1][-1]+A[0][-1])
            else:
                g[i_src-1] = -dTw
            x = lu.solve(g)       
            if abs(m) == 1: x[i_src:] += dW
            z[3,2+m,l] = x[-1]
    # Call gssh
    for i in range(0,len(thephis)):
        theta,phi = thetaphis[i]
        u,v,w0 = shtools_gssh.gssh_m2(z, theta, phi, ps[i], dps[i], lmax)
        w[-1].append(w0)
        
    Label = 'l=' + str(l) 
    pl.figure(1)
    pl.plot(PERIOD,ERROR,label=Label)
    pl.legend()
    pl.figure()
    pl.plot(r,W0.real,r,W0.imag+1.e-9,rg[1:],x.real,rg[1:],x.imag+1.e-9)
    pl.title(Label)    
#print ERROR    

pl.show()



    
