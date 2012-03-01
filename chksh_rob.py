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
#from dsm import imats,imats_new
#import dpbtf2

#General variables:
#PERIOD = [100.]#Vector with the periods
PERIOD = np.arange(10,110,1) #Vector with the periods
L      = [10, 50, 75, 100] #Vector with l
m      = 1
r_src  = 6321 

#Theoretical solution's varibles
rres   = 1.      #Resolution in km  (must agree with the source)
r      = np.arange(0.,6372., rres)
i_srf  = len(r)-1
r_srf  = r[-1]
#Numerical solution's variables
dr     = 1. # Resolution (km) above the source
nb     = 2000 #Gridpoints below the source
drmin= 1.	#  
drmax= 30	# How rapidly increases the spacing
spbelow = dr*np.linspace(drmin,drmax, nb) #spacing below
norm=np.float(r_src-1)/np.sum(spbelow)
spbelow = norm*spbelow # Normalization
rbelow = np.cumsum(spbelow[::-1])
rbelow=np.hstack(([0.], rbelow))
#print rbelow[:10]
#print rbelow[-10:]
#rbelow[-1]-rbelow[-2]
#rbelow[2]-rbelow[1]
	
#rbelow = np.linspace(0, r_src-dr, nb)

rabove = np.arange(r_src, r_srf+1,dr)
rg     = np.hstack((rbelow, rabove))




for l in L:
    
    ERROR  = []   # Resetting the vector with the errors
    
    for period in PERIOD:

        i_src  = np.int(r_src/rres)
        beta   = 4. # in km/s
        rho    = 3. # gm/cm^3 (need to account for units in source)
        #*print "r_src=%g" % r_src
        mu     = rho*beta*beta
        mu_src = rho*beta*beta 
       
        
        #
        mrt    = 1.
        mrp    = 1.
        mtt    = 1.
        mpp    = 1.
        mtp    = 1.
        Qm     = 1000.
        tau    = 1024.
        omega  = complex(2.*math.pi/period,1./tau)
        
        xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
        #xdsm   = 1.+0j
        beta   = beta*xdsm
        xdsm   = xdsm*xdsm
        mu     = mu*xdsm
        mu_src = mu_src*xdsm
        
        
        jn = np.zeros(len(r),complex)
        yn = np.zeros(len(r),complex)
        for i in range(0,len(r)):
            jn[i] = scipy.special.sph_jn(l,omega*r[i]/beta)[0][l]
        for i in range(i_src, i_srf+1):
            yn[i] = scipy.special.sph_yn(l,omega*r[i]/beta)[0][l]
        
        j_src    = jn[i_src]
        jp_src   = omega/beta*scipy.special.sph_jn(l,omega*r_src/beta)[1][l]
        j_srf    = jn[i_srf]
        jp_srf   = omega/beta*scipy.special.sph_jn(l,omega*r_srf/beta)[1][l]
        y_src    = yn[i_src]
        yp_src   = omega/beta*scipy.special.sph_yn(l,omega*r_src/beta)[1][l]
        y_srf    = yn[i_srf]
        yp_srf   = omega/beta*scipy.special.sph_yn(l,omega*r_srf/beta)[1][l]  
        
        #*print j_src,  jp_src, j_srf,   jp_srf    
        #*print y_src,  yp_src, y_srf,   yp_srf 
        
        
        b1 = math.sqrt((2*l+1)/(16.*math.pi))
        b2 = math.sqrt((2*l+1)*(l-1)*(l+2)/(64.*math.pi))
        # Set up jumps in W and T
        dW = 0.+0.j
        if (abs(m) == 1):
            dW = b1*complex(m*mrp,mrt)/(r_src*r_src*mu_src)
            #*print "dW = (%g,%g)" % (dW.real,dW.imag)
        dTw = 0.+0.j
        if (abs(m) == 2):
            dTw = b2*complex(-2*mtp,math.copysign(mpp-mtt,m))/r_src
            #*print "dTw = (%g,%g)" % (dTw.real,dTw.imag)
            
        # Linear system for boundary conditions at source depth adn surface
        #              below source     above source
        A = np.array([[ -j_src+0j ,      j_src+0j,          y_src+0j],\
                    [ -jp_src+0j ,     jp_src+0j,         yp_src+0j],\
                    [0.+0j,         jp_srf-j_srf/r_srf+0j,    yp_srf-y_srf/r_srf+0j]])
                    
        # Set boundary condition terms - why is there and r_src**2 here?              
        b = np.zeros((3,1),complex)
        b[0] = dW
        b[1] = dTw/(mu_src*r_src*r_src)
        b[2] = 0.+0j
        x = np.linalg.solve(A,b)
        #for i in range(0,3):
        #    print b[i],A[i][0]*x[0]+A[i][1]*x[1]+A[i][2]*x[2]
        
        
        # W0 is the analytical solution
        W0 = np.zeros(len(r),complex)
        # below source
        W0[0:i_src] = x[0]*jn[0:i_src]
        # Above source
        W0[i_src:] = x[1]*jn[i_src:] + x[2]*yn[i_src:]
        
        # Print some informatoin on analytical solution
        #*print "W = ",x[0]*jn[i_src],x[1]*jn[i_src]+x[2]*yn[i_src]
        #*print "Tsrf=",mu*(W0[len(r)-1]-W0[len(r)-2] - W0[len(r)-1]/r_srf)
        #*print "delta(W) = ",W0[i_src]-W0[i_src-1]
        #*print "delta(Tw) = ",mu_src*(W0[i_src+1]-2.*W0[i_src]+W0[i_src-1])
        #,W0[i_src+1]-W0[i_src],x[1]*jp_src+x[2]*yp_src,W0[i_src]-W0[i_src-1],x[0]*jp_src
        #print jn[i_src]-jn[i_src-1], jp_src,yn[i_src+1]-yn[i_src], yp_src,
        
        
        ########            Now for the DSM Solution
        ######## Function to calculate fundamental element matrices on rg[]
        ######## Not that ll matrices have the diagonal [1] index, off-diagnal in [0] index
        ########
        def Imats(rg,i_src):
        
            n   = len(rg)
            flag = rg[0] == 0.
            if flag: 
                n = len(rg)-1                
            
            A = range(0,n)
            B = range(1,n)
            C = range(0,n-1) 
            I0  = np.zeros((2,n))
            I1  = np.zeros((2,n))
            I2a = np.zeros((2,n))
            I2b = np.zeros((2,n))
            I3  = np.zeros((2,n))
            r_0 = rg[:-1]
            r_1 = rg[1:]
            DR  = r_1-r_0
	    

	  
            if flag: 
                    I0[1][A]   =  DR[A]*(DR[A]*(DR[A]*0.2 + r_0[A]*0.5) +r_0[A]*r_0[A]/3.)
                    I0[1][C]  +=  DR[B]*(DR[B]*(DR[B]*0.2 - r_1[B]*0.5) +r_1[B]*r_1[B]/3.)
                    I0[0][B]   =  DR[B]*(DR[B]*DR[B]*0.05 + r_0[B]*r_1[B]/6.)
                    #J0 = DR[i_src]*(DR[i_src]*(DR[i_src]*0.2 - r_1[i_src]*0.5) +r_1[i_src]**2/3.)
                    I1[1][A]   =  DR[A]/3. 
                    I1[1][C]  +=  DR[B]/3.
                    I1[0][B]   =  DR[B]/6.
                    #J1 = DR[i_src]/3.
                    I2a[1][A]  =  (r_0[A] + 2.*r_1[A])/6.
                    I2a[1][C] -=  (r_1[B] + 2.*r_0[B])/6.
                    I2a[0][B]  = -(r_0[B] + 2.*r_1[B])/6.               
                    #J2a = -(r_1[i_src] + 2.*r_0[i_src])/6.
                    I2b[1][A]  =  (r_0[A] + 2.*r_1[A])/6.
                    I2b[1][C] -=  (r_1[B] + 2.*r_0[B])/6.
                    I2b[0][B]  =  (r_1[B] + 2.*r_0[B])/6.                    
                    #J2b = -(r_1[i_src] + 2.*r_0[i_src])/6.
                    I3[1][A]   =  (r_1[A]*r_1[A] + r_1[A]*r_0[A] + r_0[A]*r_0[A])/(DR[A]*3.) 
                    I3[1][C]  +=  (r_1[B]*r_1[B] + r_1[B]*r_0[B] + r_0[B]*r_0[B])/(DR[B]*3.)
                    I3[0][B]   = -(r_1[B]*r_1[B] + r_1[B]*r_0[B] + r_0[B]*r_0[B])/(DR[B]*3.)
                    #J3 = (r_1[i_src]*r_1[i_src] + r_1[i_src]*r_0[i_src] + r_0[i_src]*r_0[i_src])/(DR[i_src]*3.)
                    
                    
                
                      
            else:
                    I0[1][B]   =  DR[C]*(DR[C]*(DR[C]*0.2 + r_0[C]*0.5) +r_0[C]*r_0[C]/3.)
                    I0[1][C]  +=  DR[C]*(DR[C]*(DR[C]*0.2 - r_1[C]*0.5) +r_1[C]*r_1[C]/3.)
                    I0[0][B]   =  DR[C]*(DR[C]*DR[C]*0.05 + r_0[C]*r_1[C]/6.)
                    #J0 = 0.	
                    I1[1][B]   =  DR[C]/3. 
                    I1[1][C]  +=  DR[C]/3.
                    I1[0][B]   =  DR[C]/6.
                    #J1 = 0.
                    I2a[1][B]  =  (r_0[C] + 2.*r_1[C])/6.
                    I2a[1][C] -=  (r_1[C] + 2.*r_0[C])/6.
                    I2a[0][B]  = -(r_0[C] + 2.*r_1[C])/6.               
                    #J2a = 0.
                    I2b[1][B]  =  (r_0[C] + 2.*r_1[C])/6.
                    I2b[1][C] -=  (r_1[C] + 2.*r_0[C])/6.
                    I2b[0][B]  =  (r_1[C] + 2.*r_0[C])/6.                    
                    #J2b = 0.
                    I3[1][B]   =  (r_1[C]*r_1[C] + r_1[C]*r_0[C] + r_0[C]*r_0[C])/(DR[C]*3.) 
                    I3[1][C]  +=  (r_1[C]*r_1[C] + r_1[C]*r_0[C] + r_0[C]*r_0[C])/(DR[C]*3.)
                    I3[0][B]   = -(r_1[C]*r_1[C] + r_1[C]*r_0[C] + r_0[C]*r_0[C])/(DR[C]*3.)
                    #J3 = 0.
	    
	    
            J0, J1, J2a, J2b, J3 = 0., 0., 0., 0., 0.
#
#    
#
#
#                
            return (I0,I1,I2a,I2b,I3,J0,J1,J2a,J2b,J3)         
        
#        def Imats(rg,i_src):
#        
#            n   = len(rg)
#            j   = 0
#            if rg[0] == 0.: 
#                n = len(rg)-1
#                j = -1
#            I0  = np.zeros((2,n))
#            I1  = np.zeros((2,n))
#            I2a = np.zeros((2,n))
#            I2b = np.zeros((2,n))
#            I3  = np.zeros((2,n))
#        
#            for i in range(0,len(rg)-1):
#                r_i0 = rg[i]
#                r_i1 = rg[i+1]
#                dr   = r_i1 - r_i0
#                J0 = 0.
#                J1 = 0.
#                J2a= 0.
#                J2b = 0.
#                J3 = 0.
#        # I0: r^2*X*X
#                if r_i0 != 0.:
#                    if (i == i_src): J0 = dr*(dr*(dr*0.2 - r_i1*0.5) +r_i1*r_i1/3.)
#                    I0[1][ j ]  +=  dr*(dr*(dr*0.2 - r_i1*0.5) +r_i1*r_i1/3.)
#                    I0[0][j+1]   =  dr*(dr*dr*0.05 + r_i0*r_i1/6.)
#                I0[1][j+1]   =  dr*(dr*(dr*0.2 + r_i0*0.5) +r_i0*r_i0/3.)
#        # I1: X*X 
#                if r_i0 != 0.:
#                    if (i == i_src): J1 = dr/3.
#                    I1[1][ j ]  +=  dr/3.
#                    I1[0][j+1]   =  dr/6.
#                I1[1][j+1]   =  dr/3.
#        # I2a: r*X^dot*X
#                if r_i0 != 0.:
#                    if (i == i_src): J2a = -(r_i1 + 2.*r_i0)/6.
#                    I2a[1][ j ] -=  (r_i1 + 2.*r_i0)/6.
#                    I2a[0][j+1]  = -(r_i0 + 2.*r_i1)/6.
#                I2a[1][j+1]  =  (r_i0 + 2.*r_i1)/6.
#        # I2b: r*X*X^dot
#                if r_i0 != 0.:
#                    if (i == i_src): J2b = -(r_i1 + 2.*r_i0)/6.
#                    I2b[1][ j ] -=  (r_i1 + 2.*r_i0)/6.
#                    I2b[0][j+1]  =  (r_i1 + 2.*r_i0)/6.
#                I2b[1][j+1]  =  (r_i0 + 2.*r_i1)/6.
#        # I3: r^2*X^dot*X^dot
#                if r_i0 != 0.:
#                    if (i == i_src): J3 = (r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
#                    I3[1][ j ]  +=  (r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
#                    I3[0][j+1]   = -(r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
#                I3[1][j+1]   =  (r_i1*r_i1 + r_i1*r_i0 + r_i0*r_i0)/(dr*3.)
#        # Increment column index
#                j = j+1
#                
#            return (I0,I1,I2a,I2b,I3,J0,J1,J2a,J2b,J3)         
        
      

                
        # Redefining i_src to use it in the numerical solution
        i_src    = len(rbelow)
        
        


        for i_src in range(len(rg)-1,-1,-1):
            if r_src >= rg[i_src]: break
        #*print 'source radius = %g (%g)' % (rg[i_src],r_src)
        # Calculate elemental matrices
        (I0,I1,I2a,I2b,I3,J0,J1,J2a,J2b,J3) = Imats(rg,i_src)
        # Calculate stiffness matrix (straight from paper)
        Stiff = -(omega*omega*rho*I0 - mu*((l*(l+1)-1)*I1 -I2a - I2b + I3))
        
        # Solving via  LU factorisation
        J_src = -(omega*omega*rho*J0 - mu*((l*(l+1)-1)*J1 -J2a - J2b + J3))
        ncol = Stiff.shape[1]
        # to matlab
        nelmt = 3*ncol-2
        Stiff_data = np.zeros(nelmt,complex)
        Stiff_i   = np.zeros(nelmt,int)
        Stiff_j   = np.zeros(nelmt,int)
        j = 0
        for col in range(0,ncol):
            Stiff_data[j] = Stiff[1][col]
            Stiff_i[j] = col+1
            Stiff_j[j] = col+1
            j += 1
            if col < ncol-1:
                Stiff_data[j] = Stiff[0][col+1]
                Stiff_i[j] = col+2
                Stiff_j[j] = col+1
                j +=1
                Stiff_data[j] = Stiff[0][col+1]
                Stiff_i[j] = col+1
                Stiff_j[j] = col+2
                j +=1
        #*print j,nelmt
        srfl = open('Amat_real','w')
        sifl = open('Amat_imag','w')
        ifl = open('i','w')
        jfl = open('j','w')
        for i in range(0,nelmt):
            srfl.write('%f\n' % Stiff_data[i].real)
            sifl.write('%f\n' % Stiff_data[i].imag)
            ifl.write('%f\n' % Stiff_i[i]) 
            jfl.write('%f\n' % Stiff_j[i])        
        srfl.close()
        sifl.close()
        ifl.close()
        jfl.close()
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
        #*print Stiff_data.shape,Stiff_ij.shape
        Stiff_csc = csc_matrix((Stiff_data,Stiff_ij))
        #
        
        
        #C = scipy.linalg.cholesky_banded(Stiff)
        #print C
        
        #F1 = analyze(Stiff_csc,mode="simplicial")
        #F2 = F1.cholesky(Stiff_csc)
        # Superlu
        lu = linalg.splu(Stiff_csc)
        #sys.exit(0)
        
        
        
        ncol = Stiff.shape[1]
        g = np.zeros((ncol),complex)
        if abs(m) == 1:
            (J0,J1,J2a,J2b,J3,K0,K1,K2a,K2b,K3) = Imats(rg[i_src:],i_src)
            A = -(omega*omega*rho*J0 - mu*((l*(l+1)-1)*J1 -J2a - J2b + J3))
            g[i_src-1] = J_src
        #-dW*(A[1][0]+A[0][1])
            for j in range (i_src,ncol-1):
                k = j-i_src+1
                g[j] = -dW*(A[0][k]+A[1][k]+A[0][k+1])
            g[ncol-1] = -dW*(A[1][A.shape[1]-1]+A[0][A.shape[1]-1])
        else:
            g[i_src-1] = -dTw
        #x = scipy.linalg.cho_solve_banded((C,False),g)
        # Matlab
        gfl = open('g','w')
        for i in range(0,ncol):
            gfl.write('%f\n' % g[i].real)        
        gfl.close()
        
        
        # Solve for displacement vector
        x = lu.solve(g)
        #x = linalg.spsolve(Stiff_csc,g)
        
        if abs(m) == 1: x[i_src-1:] += dW
        
        
        # All done! print out some statistics
        #print "W=",x[i_src-2],x[i_src-1]
        #print len(rg)
        #print "Tsrf=",mu*(x[len(rg)-2]-x[len(rg)-3] - x[len(rg)-2]/r_srf)
        #print "delta(W) =",x[i_src-1]-x[i_src-2]
        #print "delta(Tw) = ",mu_src*(x[i_src]-2.*x[i_src-1]+x[i_src-2])
        #End statistics
        
        #Plot the solutions to compare the results.
        #pl.plot(r,W0.real,r,W0.imag+1.e-9,rg[1:],x.real,rg[1:],x.imag+1.e-9)
        #pl.show()
        #End plots
        
        
        #Compute the relative error at surface:
        error  = np.absolute(W0[-1]-x[-1])/np.absolute(W0[-1]) * 100.
        ERROR[len(ERROR):] = [error]  
    
    Label = 'l=' + str(l) 
    pl.figure(1)
    pl.plot(PERIOD,ERROR,label=Label)
    pl.legend()
    pl.figure()
    pl.plot(r,W0.real,r,W0.imag+1.e-9,rg[1:],x.real,rg[1:],x.imag+1.e-9)
    pl.title(Label)    
#print ERROR    
#pl.figure()
pl.show()



    
