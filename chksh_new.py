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
#import dpbtf2

def imats_nnew(rg,i_src):
    r1 = rg[1:]
    r0 = rg[0:-1]
    #print r0[i_src-1],r1[i_src-1]
    dr = r1-r0
    tmp12 = np.zeros(len(dr)+1)
    tmp11 = np.zeros(len(dr)+1)
    tmp22 = np.zeros(len(dr)+1)

    # I0: r^2*X*X
    tmp12[1:]   = dr*(dr*dr*0.05 + r0*r1/6.) 
    tmp11[0:-1] = dr*(dr*(dr*0.2 - r1*0.5) +r1*r1/3.) 
    tmp22[1:]   = dr*(dr*(dr*0.2 + r0*0.5) +r0*r0/3.)
    J0 = tmp22[i_src]
    I0 = np.array([tmp12,(tmp11+tmp22)])
    # I1: X*X 
    tmp11[0:-1] =  dr/3.
    tmp12[1:]   =  dr/6.
    tmp22[1:]   =  dr/3.
    J1 = tmp22[i_src]
    I1 = np.array([tmp12,(tmp11+tmp22)])
    # I2a: r*X^dot*X
    tmp11[0:-1] = -(r1 + 2.*r0)/6. 
    tmp12[1:]   = -(r0 + 2.*r1)/6. 
    tmp22[1:]   =  (r0 + 2.*r1)/6.
    J2a = tmp22[i_src]
    I2a = np.array([tmp12,(tmp11+tmp22)])
    # I2b: r*X*X^dot
    tmp11[0:-1] = -(r1 + 2.*r0)/6.
    tmp12[1:]   =  (r1 + 2.*r0)/6. 
    tmp22[1:]   =  (r0 + 2.*r1)/6.
    J2b = tmp22[i_src]
    I2b = np.array([tmp12,(tmp11+tmp22)])
    # I3: r^2*X^dot*X^dot
    tmp11[0:-1] =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.) 
    tmp12[1:]   = -(r1*r1 + r1*r0 + r0*r0)/(dr*3.) 
    tmp22[1:]   =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.)
    J3 = tmp22[i_src]
    I3 = np.array([tmp12,(tmp11+tmp22)])

    return (I0,I1,I2a,I2b,I3,J0,J1,J2a,J2b,J3)
    



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


#PERIOD = [100.]#Vector with the periods
PERIOD = np.arange(10,1000,100) #Vector with the periods

L      = [100] #, 50, 75, 100] #Vector with l
m      = 2
r_src  = 6321 

#Theoretical solution's variables
rres   = 1.      #Resolution in km  (must agree with the source)
r      = np.arange(0.,6372., rres)
i_srf  = len(r)-1
r_srf  = r[-1]
#Numerical solution's variables
dr     = 1. # Resolution (km) above the source
nb     = 5000 #Gridpoints below the source

#drmin= 1.	#  
#drmax= 30	# How rapidly increases the spacing
#spbelow = dr*np.linspace(drmin,drmax, nb) #spacing below
#norm=np.float(r_src-1)/np.sum(spbelow)
#spbelow = norm*spbelow # Normalization
#rbelow = np.cumsum(spbelow[::-1])
#rbelow=np.hstack(([0.], rbelow))
	
rbelow = np.linspace(0, r_src-dr, nb)
rabove = np.arange(r_src, r_srf+1,dr)
rg     = np.hstack((rbelow, rabove))

###
figprops = dict(figsize=(30., 15 / 1.618))
adjustprops = dict(left=0.05, bottom=0.05, right=0.9, top=0.93, wspace=0.4, hspace=0.4)      # Subplot properties
fig = pl.figure(**figprops)                                                              # New figure
nxpl = int((len(L)+2)/2.+0.51)
ipl = 1
###

for l in L:
    LL=np.sqrt(l*(l+1))
    
    ERROR0  = []   # Resetting the vector with the errors
    ERROR1  = []   # Resetting the vector with the errors
    
    for period in PERIOD:

        i_src  = np.int(r_src/rres)
 
        omega  = complex(2.*math.pi/period,1./tau)        
        if period > 900.:
            print 'frequency 0'
            omega  = complex(0.,1./tau)        
            #xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
        xdsm   = 1.+0j
        beta_dsm   = beta*xdsm
        xdsm   = xdsm*xdsm
        mu_dsm     = mu*xdsm
        mu_src_dsm = mu_src*xdsm
        
        
        jn = np.zeros(len(r),complex)
        yn = np.zeros(len(r),complex)
#        vecsph_jn=np.vectorize(scipy.special.sph_jn, otypes=[np.complex])
#        vecsph_yn=np.vectorize(scipy.special.sph_yn, otypes=[np.complex])
#        jn = vecsph_jn(l,omega*r/beta)[0][l]
#        yn = vecsph_yn(l,omega*r/beta)[0][l]
        
        for i in range(0,len(r)):
            jn[i] = scipy.special.sph_jn(l,omega*r[i]/beta_dsm)[0][l]
        for i in range(i_src, i_srf+1):
            yn[i] = scipy.special.sph_yn(l,omega*r[i]/beta_dsm)[0][l]
        
            #print 'Analytical: i_src = %d, r_src = %g' % (i_src,r[i_src])

        j_src    = jn[i_src]
        jp_src   = omega/beta_dsm*scipy.special.sph_jn(l,omega*r_src/beta_dsm)[1][l]
        j_srf    = jn[i_srf]
        jp_srf   = omega/beta_dsm*scipy.special.sph_jn(l,omega*r_srf/beta_dsm)[1][l]
        y_src    = yn[i_src]
        yp_src   = omega/beta_dsm*scipy.special.sph_yn(l,omega*r_src/beta_dsm)[1][l]
        y_srf    = yn[i_srf]
        yp_srf   = omega/beta_dsm*scipy.special.sph_yn(l,omega*r_srf/beta_dsm)[1][l]    
        
        b1 = math.sqrt((2*l+1)/(16.*math.pi))
        b2 = math.sqrt((2*l+1)*(l-1)*(l+2)/(64.*math.pi))
        # Set up jumps in W and T
        dW = 0.+0.j
        if (abs(m) == 1):
            dW = b1*complex(m*mrp,mrt)/(r_src*r_src*mu_src_dsm)
            #*print "dW = (%g,%g)" % (dW.real,dW.imag)
        dTw = 0.+0.j
        if (abs(m) == 2):
            dTw = b2*complex(-2*mtp,math.copysign(mpp-mtt,m))/r_src
            #*print "dTw = (%g,%g)" % (dTw.real,dTw.imag)
            
        # Linear system for boundary conditions at source depth adn surface
        #              below source     above source
        A = np.array([[  -j_src ,        j_src,          y_src],\
                      #[ -jp_src ,       jp_src,          yp_src],\
                      [ j_src/r_src-jp_src+0j ,     jp_src-j_src/r_src+0j,    yp_src-y_src/r_src+0j],\
                      [  0.+0j  , jp_srf-j_srf/r_srf,    yp_srf-y_srf/r_srf]])
                    
        # Set boundary condition terms - why is there and r_src**2 here?              
        b = np.zeros((3,1),complex)
        b[0] = dW
        b[1] = dTw/(mu_src_dsm*r_src*r_src)
        b[2] = 0.+0j
        x = np.linalg.solve(A,b)
        print 'dW = ',dW,' = ',x[2]*y_src+x[1]*j_src-x[0]*j_src
        
        # W0 is the analytical solution
        W0 = np.zeros(len(r),complex)
        # below source
        W0[0:i_src] = x[0]*jn[0:i_src]
        # Above source
        W0[i_src:] = x[1]*jn[i_src:] + x[2]*yn[i_src:]
        
      
        
        ########            Now for the DSM Solution
        ######## Function to calculate fundamental element matrices on rg[]
        ######## Not that ll matrices have the diagonal [1] index, off-diagnal in [0] index
        ########

        # Redefining i_src to use it in the numerical solution
        i_src    = len(rbelow)

        #print 'DSM       : i_src = %d, r_src = %g' % (i_src,rg[i_src])

        for i_src in range(len(rg)-1,-1,-1):
            if r_src >= rg[i_src]: break

        # Calculate elemental matrices
###########  ###### 

	(K0,K1,K2a,K2b,K3,J0,J1,J2a,J2b,J3) = imats_nnew(rg,i_src)
	Stiff = -(omega*omega*rho*K0 - mu_dsm*((l*(l+1)-1)*K1 - K2a - K2b + K3))
        # Set boundary condition of zero at r=0 by removing first row & column of Stiff
        Stiff = Stiff[:,1:]
	Source_correction = -(omega*omega*rho*J0 - mu_dsm*((l*(l+1)-1)*J1 -J2a - J2b + J3))
        
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

        #####

	ncol = Stiff.shape[1]
	g0 = np.zeros((ncol),complex)
	A0 = Stiff[:,i_src-1:]
	if abs(m) == 1:
          print l,m,period,A0[0][0],A0[1][1],A0[0][1],A0[0][0]+A0[1][1]+A0[0][1]
          g0[i_src-1:-1] =  -dW*(A0[0][:-1]+A0[1][:-1]+A0[0][1:])
	  g0[-1] = -dW*(A0[1][-1]+A0[0][-1])
	else:
	    g0[i_src-1] = -dTw
	x0 = lu.solve(g0)       
        if abs(m) == 1: x0[i_src-1:] += dW

        #Compute the relative error at surface:
        error  = np.absolute(W0[-1]-x0[-1])/np.absolute(W0[-1]) * 100.
        ERROR0[len(ERROR0):] = [error]

        #####
	ncol = Stiff.shape[1]
	g1 = np.zeros((ncol),complex)
	#A = Stiff[:,i_src:]
        #print l,m,period, A[1][0],A[0][0],Source_correction
        #A[1][0] -= Source_correction
        #A[0][0] = 0.+0.j
        (K0,K1,K2a,K2b,K3,J0,J1,J2a,J2b,J3) = imats_nnew(rg[i_src:],0)
	A1 = -(omega*omega*rho*K0 - mu_dsm*((l*(l+1)-1)*K1 -K2a - K2b + K3))
	if abs(m) == 1:
          print l,m,period,A1[1][0],A1[0][1],K0[1][0],K0[0][1],K1[1][0],K1[0][1],K2a[1][0],K2a[0][1],K2b[1][0],K2b[0][1],K3[1][0],K3[0][1]
          print l,m,period,A1[1][0]+A1[0][1]
          g1[i_src-1]  = -dW*(A1[1][0]+A1[0][1])
	  g1[i_src:-1] = -dW*(A1[0][1:-1]+A1[1][1:-1]+A1[0][2:])
	  g1[-1]       = -dW*(A1[0][-1]+A1[1][-1])
	else:
	    g1[i_src-1] = -dTw
	x1 = lu.solve(g1)       
        if abs(m) == 1: x1[i_src-1:] += dW

       
        #Compute the relative error at surface:
        error  = np.absolute(W0[-1]-x1[-1])/np.absolute(W0[-1]) * 100.
        ERROR1[len(ERROR1):] = [error]

        #####
        
    
    Label = 'l=' + str(l) 
    ax = fig.add_subplot(2,nxpl,len(L)+1)
    pl.figure(1)
    pl.plot(PERIOD,ERROR0,label=Label)
    pl.legend()

    bx = fig.add_subplot(2,nxpl,len(L)+2)
    pl.plot(PERIOD,ERROR1,label=Label)
    pl.legend()

    cx = fig.add_subplot(2,nxpl,ipl)
    pl.plot(r,W0.real,r,W0.imag+1.e-9,rg[1:],x1.real,rg[1:],x1.imag+1.e-9)
    pl.title(Label)
    ipl += 1
    
#print ERROR    

pl.show()



    
