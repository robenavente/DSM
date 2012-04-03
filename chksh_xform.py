#!/usr/bin/env python
import sys
import os
import scipy.special
import scipy.linalg
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from scipy.lib.lapack import get_lapack_funcs
from scipy.sparse import csc_matrix,linalg
import dsm #,imats_new
import shtools_gssh
import pickle

#General variables:

mrt    = 1.
mrp    = 1.
mtt    = 1.
mpp    = 1.
mtp    = 1.
Qm     = 2000.
tau    = 4096.

beta   = 4. # in km/s
rho    = 3. # gm/cm^3 (need to account for units in source)
#*print "r_src=%g" % r_src
mu     = rho*beta*beta
mu_src = rho*beta*beta     

lmax   = 500
f      = np.arange(0.00025,0.0251,0.00025) #Vector with the frequencies
PERIOD = 1./f #Vector with the periods
L      = range(1,lmax+1) #Vector with l
phi = 25.
THETA  = np.arange(10.,91.,10.)
thetaphis =zip(THETA,phi*np.ones(len(THETA)))
#ps = []
#dps = []
#for theta,phi in thetaphis:
#    z = math.cos(theta*math.pi/180.)
#    p,dp = shtools_gssh.plmbar_d1_m2(lmax, z)
#    ps.append(p)
#    dps.append(dp)
plms = []
for theta,phi in thetaphis:
    z = math.cos(theta*math.pi/180.)
    p,dp = shtools_gssh.plmbar_d1_m2(lmax, z)
    plms.append( (theta,phi,p,dp))

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
(K0,K1,K2a,K2b,K3) = dsm.imats(rg,i_src)
for period in PERIOD:

    w.append([])
    omega  = complex(2.*math.pi/period,1./tau)        
    xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
    #xdsm   = 1.+0j
    beta_dsm   = beta*xdsm
    xdsm   = xdsm*xdsm
    mu_dsm     = mu*xdsm
    mu_src_dsm = mu_src*xdsm

    z = np.zeros((3,5,lmax+1),complex)
    for l in L:
    
        ERROR  = []   # Resetting the vector with the errors
                
        ########            Now for the DSM Solution
        ######## Function to calculate fundamental element matrices on rg[]
        ######## Not that ll matrices have the diagonal [1] index, off-diagnal in [0] index
        ########


        # Calculate elemental matrices
###########  ###### 

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
        for m in range(-2,3,1):
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
            z[2,2+m,l] = x[-1]
    # Call gssh
    #    for i in range(0,len(thetaphis)):
    #    theta,phi = thetaphis[i]
    #    u,v,w0 = shtools_gssh.gssh_m2(z, theta, phi, ps[i], dps[i], lmax)
    #    w[-1].append(w0)
    for theta,phi,p,dp in plms:
        u,v,w0 = shtools_gssh.gssh_m2(z, theta, phi, p, dp, lmax)
        w[-1].append(w0)
        
outfile = open('w.pkl', 'wb')
pickle.dump(w, outfile)
outfile.close()
Label = 'Theta = %g Phi = %g' % thetaphis[0]
w = np.array(w).transpose()
zrs = np.zeros((len(thetaphis),1),complex)
y=np.conjugate(np.append(zrs,w,1))
nt = 2*(y.shape[1]-1)
g=np.zeros((len(thetaphis),nt))
t = 0.5*PERIOD[-1]*np.arange(0.,nt)
for i in range(0,len(thetaphis)):
    g[i,:] = fft.irfft(y[i])
    for j in range(0,nt):
        g[i,j] *= math.exp(t[i]/tau)
plt.figure()
plt.hold(True)
tt0 = np.zeros((2,len(thetaphis)))
tt1 = np.zeros((2,len(thetaphis)))
for i in range(0,len(thetaphis)):
    theta,phi = thetaphis[i]
    tt0[0][i] = theta
    tt1[0][i] = theta
    x1 = 6371.*math.sin(0.5*math.pi*theta/180.)
    x2 = 6371.*math.cos(0.5*math.pi*theta/180.)
    x3 = math.sqrt(x1**2+(6371.-x2)**2)
    tt0[1][i] = (2.*x1/beta) % PERIOD[0]
    tt1[1][i] = (2.*x3/beta) % PERIOD[0]
    plt.plot(t,4.*g[i]/g[i].max()+theta)
plt.plot(tt0[1,:],tt0[0,:],tt1[1,:],tt1[0,:])
plt.savefig('chksh_xform.pdf')
plt.show()
plt.hold(False)




    
