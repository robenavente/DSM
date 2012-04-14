#!/usr/bin/env python

import sys
import os
import scipy.special
import scipy.linalg
import scipy.misc
import math
import pylab as pl
import numpy as np
from scipy.lib.lapack import get_lapack_funcs
from scipy.sparse import csc_matrix,linalg
from dsm import * #,imats_new
#import dpbtf2

#General variables:
def main():
    global LL, lbd_dsm, mu_dsm,alpha_dsm, beta_dsm, omega, lbd_dsm
    mrr    = 1.
    mrt    = 1.
    mrp    = 1.
    mtt    = 1.
    mpp    = 1.
    mtp    = 1.
    Qm     = 1000.
    tau    = 1024.

    beta   = 4. # in km/s
    rho    = 3. # gm/cm^3 (need to account for units in source)
    alpha  = 7. # km/s
    #*print "r_0=%g" % r_0
    mu     = rho*beta*beta
    mu_0   = rho*beta*beta
    lbd    = alpha*alpha*rho-2*mu
    lbd_0  = alpha*alpha*rho-2*mu




    PERIOD = [100.]#Vector with the periods
    #PERIOD = np.arange(10,100,1) #Vector with the periods
    L = [75]
    #L      = [10, 50, 75, 100] #Vector with l
    m      = 1
    r_0  = 6321

    #Theoretical solution's variables
    rres   = 1.      #Resolution in km  (must agree with the source)
    r      = np.arange(0.,6372., rres)
    i_a  = len(r)-1
    r_a  = r[-1]
    #Numerical solution's variables
    dr     = 1. # Resolution (km) above the source
    nb     = 5000 #Gridpoints below the source

    #drmin= 1.    #
    #drmax= 30    # How rapidly increases the spacing
    #spbelow = dr*np.linspace(drmin,drmax, nb) #spacing below
    #norm=np.float(r_0-1)/np.sum(spbelow)
    #spbelow = norm*spbelow # Normalization
    #rbelow = np.cumsum(spbelow[::-1])
    #rbelow=np.hstack(([0.], rbelow))

    rbelow = np.linspace(0, r_0-dr, nb)
    rabove = np.arange(r_0, r_a+1,dr)
    rg     = np.hstack((rbelow, rabove))
     # Calculate elemental matrices
    (K0,K1,K2a,K2b,K3) = imats(rg,len(rbelow))
    
    

    for l in L:

        ERRORu   = []   # Resetting the vector with the errors
        ERRORv   = []   
        ERRORw   = []
        
        LL = np.sqrt(l*(l+1))
        for period in PERIOD:

            i_0  = np.int(r_0/rres)

            omega  = complex(2.*math.pi/period,1./tau)
            xdsm   = complex(1.0+math.log(0.5*omega.real/math.pi)/(math.pi*Qm),-0.5/Qm);
            #xdsm   = 1.+0j
            beta_dsm   = beta*xdsm
            alpha_dsm  = alpha*xdsm
            xdsm   = xdsm*xdsm
            mu_dsm     = mu*xdsm
            mu_0_dsm = mu_0*xdsm
            lbd_dsm =     lbd*xdsm
            lbd_0_dsm = lbd_0*xdsm


            jn = np.zeros(len(r),complex)
            yn = np.zeros(len(r),complex)
            jna = np.zeros(len(r),complex)
            yna = np.zeros(len(r),complex)
            jp = np.zeros(len(r),complex)
            yp = np.zeros(len(r),complex)
            jpa = np.zeros(len(r),complex)
            ypa = np.zeros(len(r),complex)            
            
    #        vecsph_jn=np.vectorize(scipy.special.sph_jn, otypes=[np.complex])
    #        vecsph_yn=np.vectorize(scipy.special.sph_yn, otypes=[np.complex])
    #        jn = vecsph_jn(l,omega*r/beta)[0][l]
    #        yn = vecsph_yn(l,omega*r/beta)[0][l]

            for i in range(0,len(r)):
                jn[i]  = scipy.special.sph_jn(l,omega*r[i]/beta_dsm)[0][l]
                jna[i] = scipy.special.sph_jn(l,omega*r[i]/alpha_dsm)[0][l]
                jp[i]  = omega/beta_dsm*scipy.special.sph_jn(l,omega*r[i]/beta_dsm)[1][l]
                jpa[i] = omega/alpha_dsm*scipy.special.sph_jn(l,omega*r[i]/alpha_dsm)[1][l]
                
            for i in range(i_0, i_a+1):
                yn[i]  = scipy.special.sph_yn(l,omega*r[i]/beta_dsm)[0][l]
                yna[i] = scipy.special.sph_yn(l,omega*r[i]/alpha_dsm)[0][l]  
                yp[i]  = omega/beta_dsm*scipy.special.sph_yn(l,omega*r[i]/beta_dsm)[1][l]
                ypa[i] = omega/alpha_dsm*scipy.special.sph_yn(l,omega*r[i]/alpha_dsm)[1][l]
               

            j_0    = jn[i_0]
            ja_0   = jna[i_0]
            jp_0   = jp[i_0]
            jpa_0  = jpa[i_0]
            j_a    = jn[i_a]
            ja_a   = jna[i_a]
            jp_a   = jp[i_a]
            jpa_a  = jpa[i_a]
            y_0    = yn[i_0]
            ya_0   = yna[i_0]
            yp_0   = yp[i_0]
            ypa_0  = ypa[i_0]
            y_a    = yn[i_a]
            ya_a   = yna[i_a]
            yp_a   = yp[i_a]
            ypa_a  = ypa[i_a]

            b1 = math.sqrt((2*l+1)/(16.*math.pi))
            b2 = math.sqrt((2*l+1)*(l-1)*(l+2)/(64.*math.pi))
            # Set up jumps in W and T
            dW = 0.+0.j
            if (abs(m) == 1):
                dW = b1*complex(m*mrp,-mrt)/(r_0*r_0*mu_0_dsm)
                #*print "dW = (%g,%g)" % (dW.real,dW.imag)
            dTw = 0.+0.j
            if (abs(m) == 2):
                dTw = b2*complex(-2*mtp,math.copysign(-mpp+mtt,m))/r_0
                #*print "dTw = (%g,%g)" % (dTw.real,dTw.imag)

            # Linear system for boundary conditions at source depth adn surface
            #              below source     above source
            A = np.array([[ -j_0+0j ,      j_0+0j,          y_0+0j],\
                       # [ j_0/r_0-jp_0+0j ,     jp_0-j_0/r_0+0j,    yp_0-y_0/r_0+0j],\
                         [ -jp_0+0j ,     jp_0+0j,         yp_0+0j],\
                          [0.+0j,         jp_a-j_a/r_a+0j,    yp_a-y_a/r_a+0j]])

           # Set boundary condition terms
            b = np.zeros((3,1),complex)
            ####Check
#            dW=0            
            
            
            b[0] = dW
            b[1] = dTw/(mu_0_dsm*r_0*r_0)
            b[2] = 0.+0j
            x = np.linalg.solve(A,b)

            # W0 is the analytical solution
            W0 = np.zeros(len(r),complex)
            # below source
            W0[0:i_0] = x[0]*jn[0:i_0]
            # Above source
            W0[i_0:] = x[1]*jn[i_0:] + x[2]*yn[i_0:]

            ####Solving for S and T 
            ###Jumps:

            dU, dRu, dV, dSv = np.zeros(4, 'complex')
            aux = lbd_0_dsm-2*mu_0_dsm
            
            if m!= 0: msgn = m/abs(m) 
            
            if (m==0):
                dU  += 2*b1*mrr/aux/(r_0**2)
                dRu += 2*b1*(mtt+mpp-2*lbd_0_dsm*mrr/aux)/r_0
                dSv += b1*LL*(-mtt-mpp+2*mrr*lbd_0_dsm/aux)/r_0
            elif(abs(m)==1):
                dV  += b1*complex(-msgn*mrt, mrp)/mu_0_dsm/(r_0**2)                
            elif(abs(m)==2):
                dSv += -b2*complex(mtt-mpp, 2*msgn*mtp)/r_0
            ##### Coeff (Matrix)
           #Def  each row

            row1 = 1./np.sqrt(rho)*np.array(\
            [jpa_0, ypa_0, -jpa_0, LL*LL/r_0*j_0,LL*LL/r_0*y_0,-LL*LL/r_0*j_0])

            row2 = LL/np.sqrt(rho)/r_0*np.array(\
            [ja_0, ya_0, -ja_0, j_0+r_0*jp_0, y_0+r_0*yp_0, -j_0-r_0*jp_0])

            row3 = 2*mu_0_dsm*LL/r_0/np.sqrt(rho)*np.array(\
            [gS1(r_0)*ja_0+gS2(r_0)*jpa_0, gS1(r_0)*ya_0+gS2(r_0)*ypa_0,\
            -gS1(r_0)*ja_0-gS2(r_0)*jpa_0, j_0*gS3(r_0)+jp_0*gS4(r_0),\
            y_0*gS3(r_0)+yp_0*gS4(r_0), -j_0*gS3(r_0)-jp_0*gS4(r_0) ])

            row4 = 1./np.sqrt(rho)*np.array(\
            [gR1(r_0)*ja_0+gR2(r_0)*jpa_0, gR1(r_0)*ya_0+gR2(r_0)*ypa_0,\
            -gR1(r_0)*ja_0-gR2(r_0)*jpa_0, j_0*gR3(r_0)+jp_0*gR4(r_0),\
            y_0*gR3(r_0)+yp_0*gR4(r_0), -j_0*gR3(r_0)-jp_0*gR4(r_0) ])

            row5 = np.array(\
            [gS1(r_a)*ja_a+gS2(r_a)*jpa_a, gS1(r_a)*ya_a+gS2(r_a)*ypa_a,\
            0.+0.j                       , j_a*gS3(r_a)+jp_a*gS4(r_a),\
            y_a*gS3(r_a)+yp_a*gS4(r_a)   , 0.+0.j ])

            row6 = np.array(\
            [gR1(r_a)*ja_a+gR2(r_a)*jpa_a, gR1(r_a)*ya_a+gR2(r_a)*ypa_a,\
            0.+0.j                       , j_a*gR3(r_a)+jp_a*gR4(r_a),\
            y_a*gR3(r_a)+yp_a*gR4(r_a)   , 0.+0.j ])

            B = np.array([row1, row2, row3, row4, row5, row6])
            
            #################Just to check
            #dU = dV = 0.+0.j
            ################Justo to check

            b = np.zeros((6,1),complex)
            b[0] = dU
            b[1] = dV
            b[2] = dSv/r_0**2
            b[3] = dRu/r_0**2
           
            
            #print l, period   
            x2 = np.linalg.solve(B,b)
            #x2 = np.zeros((6,1),complex)
            ############ Calculating the Displacement from de potentials:
            U0 = np.zeros(len(r),complex)
            V0 = np.zeros(len(r),complex)
            ##Below the source
            U0[:i_0] = 1./np.sqrt(rho)*(x2[2]*jpa[:i_0]+LL*LL/r[:i_0]*x2[5]*jn[:i_0])
            V0[:i_0] = LL/np.sqrt(rho)/r[:i_0]*(x2[2]*jna[:i_0]+x2[5]*jn[:i_0]\
                     + r[:i_0]*(x2[5]*jp[:i_0]))
            ##Above the source
            U0[i_0:] = 1./np.sqrt(rho)*(x2[0]*jpa[i_0:]+x2[1]*ypa[i_0:]\
                     +LL*LL/r[i_0:]*(x2[3]*jn[i_0:]+x2[4]*yn[i_0:]))
                     
            V0[i_0:] = LL/np.sqrt(rho)/r[i_0:]*(x2[0]*jna[i_0:]+x2[1]*yna[i_0:]\
                     + x2[3]*jn[i_0:]+x2[4]*yn[i_0:]\
                     + r[i_0:]*(x2[3]*jp[i_0:]+x2[4]*yp[i_0:]))
            
            


            ########            Now for the DSM Solution
            ######## Function to calculate fundamental element matrices on rg[]
            ######## Not that all matrices have the diagonal [1] index, off-diagnal in [0] index
            ########

            # Redefining i_0 to use it in the numerical solution
            i_0    = len(rbelow)

            for i_0 in range(len(rg)-1,-1,-1):
                if r_0 >= rg[i_0]: break

           
    ###########  ######hm

            

            #p=p'=1
            Stiff1 = -omega*omega*rho*K0 + lbd_dsm*(4*K1+2*(K2a+K2b)+K3)\
                     +mu_dsm*(K1*(LL*LL+4)+2*K3)
            #p=p'=2
            Stiff2 = -omega*omega*rho*K0 + lbd_dsm*LL*LL*K1\
                     +mu_dsm*((2*LL*LL-1)*K1-K2a-K2b+K3)
	    #p=1,p'=2
            Stiff3 =  -lbd_dsm*(2*LL*K1+LL*K2b)-mu_dsm*(3*LL*K1-LL*K2a)
            #p=2,p'=1          
            Stiff4 =  -lbd_dsm*(2*LL*K1+LL*K2a)-mu_dsm*(3*LL*K1-LL*K2b)
            #P'=P=3            
            Stiff = -omega*omega*rho*K0 + mu_dsm*((LL*LL-1)*K1 -K2a - K2b + K3)           
      
            #######Checking   
#            Stiff3 = -Stiff3[:]
#            Stiff4 = -Stiff4[:]
            #Stiff2[:,:] = 0.+0j
            #print Stiff1
            #######            
            
            ncol = Stiff.shape[1]
            # to csc
            Stiffpsv_data = np.zeros(7*2*ncol-12-(2*ncol-4), complex)
            Stiffpsv_ij   = np.zeros((2,7*2*ncol-12-(2*ncol-4)))
            # Filling the diagonal:
            Stiffpsv_ij[0,:2*ncol] = Stiffpsv_ij[1,:2*ncol]=range(0,2*ncol)   
            Stiffpsv_data[:2*ncol:2] = Stiff1[1,:]
            Stiffpsv_data[1:2*ncol:2] = Stiff2[1,:]
            #Filling the 3rd-upper diagonal:
            Stiffpsv_ij[0,2*ncol:4*ncol-2] = range(0,2*ncol-2)   
            Stiffpsv_ij[1,2*ncol:4*ncol-2] = range(2,2*ncol)
            Stiffpsv_data[2*ncol:4*ncol-2:2] = Stiff1[0,1:]
            Stiffpsv_data[2*ncol+1:4*ncol-2:2] = Stiff2[0,1:]
            #Filling the 3rd-down diagonal:
            Stiffpsv_ij[0,4*ncol-2:6*ncol-4] = range(2,2*ncol)  
            Stiffpsv_ij[1,4*ncol-2:6*ncol-4] = range(0,2*ncol-2) 
            Stiffpsv_data[4*ncol-2:6*ncol-4:2] = Stiff1[0,1:]
            Stiffpsv_data[4*ncol-1:6*ncol-4:2] = Stiff2[0,1:]
            #Filling the 2nd-upper diagonal:
            Stiffpsv_ij[0,6*ncol-4:8*ncol-5] = range(0,2*ncol-1)    
            Stiffpsv_ij[1,6*ncol-4:8*ncol-5] = range(1,2*ncol) 
            Stiffpsv_data[6*ncol-4:8*ncol-5:2]= Stiff4[1,:]
            Stiffpsv_data[6*ncol-3:8*ncol-5:2]= Stiff3[0,1:]       
            #Filling the 2nd-down diagonal:
            Stiffpsv_ij[0,8*ncol-5:10*ncol-6] =  range(1,2*ncol)   
            Stiffpsv_ij[1,8*ncol-5:10*ncol-6] =  range(0,2*ncol-1)
            Stiffpsv_data[8*ncol-5:10*ncol-6:2]= Stiff4[1,:]
            Stiffpsv_data[8*ncol-4:10*ncol-6:2]= Stiff3[0,1:]   
            #Filling the 4th-upper diagonal:
            Stiffpsv_ij[0,10*ncol-6:11*ncol-7] =  range(0,2*ncol-3,2)   
            Stiffpsv_ij[1,10*ncol-6:11*ncol-7] =  range(3,2*ncol,2)
            Stiffpsv_data[10*ncol-6:11*ncol-7]=   Stiff4[0,1:]
           #Filling the 4th-lower diagonal:
            Stiffpsv_ij[0,11*ncol-7:12*ncol-8] = range(3,2*ncol,2)
            Stiffpsv_ij[1,11*ncol-7:12*ncol-8] = range(0,2*ncol-3,2)  
            Stiffpsv_data[11*ncol-7:12*ncol-8]=  Stiff4[0,1:]
	  
#            Ind =(Stiffpsv_ij[0,:] == 2*ncol-2) * (Stiffpsv_ij[1,:] == 2*ncol-1)
#            Ind = np.nonzero(Ind)[0]
#            print Stiffpsv_data[Ind]
            Stiffpsv_csc = csc_matrix((Stiffpsv_data,Stiffpsv_ij))  
            
            
             
            #Checking the final shape with a picture:
#            AStiffpsv_csc= Stiffpsv_csc.todense()                      
#            scipy.misc.imsave('spheroidalMatrix.png',1*(AStiffpsv_csc != 0.+0.j))            
                
        # to csc
            Stiff_data = np.zeros((3*ncol-2),complex)
            Stiff_ij   = np.zeros((2,3*ncol-2))
            
            ##Diagonal
            Stiff_data[:ncol] = Stiff[1,:]               
            Stiff_ij[0,:ncol] = Stiff_ij[1,:ncol] = range(0,ncol)
            ##Super Diagonal
            Stiff_data[ncol:2*ncol-1] = Stiff[0,1:]
            Stiff_ij[0,ncol:2*ncol-1] = range(0,ncol-1) 
            Stiff_ij[1,ncol:2*ncol-1] = range(1,ncol)
            ##Inferior Diagonal
            Stiff_data[2*ncol-1:3*ncol-2] = Stiff[0,1:]
            Stiff_ij[0,2*ncol-1:3*ncol-2] = range(1,ncol) 
            Stiff_ij[1,2*ncol-1:3*ncol-2] = range(0,ncol-1)
         
#            j = 0
#            for col in range(0,ncol):
#                Stiff_data[j] = Stiff[1][col]
#                Stiff_ij[0,j] = col
#                Stiff_ij[1,j] = col
#                j += 1
#        
#                if col < ncol-1:
#                    Stiff_data[j] = Stiff[0][col+1]
#                    Stiff_ij[0,j] = col
#                    Stiff_ij[1,j] = col+1
#                    j += 1
#                    Stiff_data[j] = Stiff[0][col+1]
#                    Stiff_ij[0,j] = col+1
#                    Stiff_ij[1,j] = col
#                    j += 1
        
            Stiff_csc = csc_matrix((Stiff_data,Stiff_ij))
            
            #Building the full matrix:
            #print np.shape(Stiff_ij)    
            #Stiffsh_ij = np.array([Stiff_ij[0][:]+2*ncol,Stiff_ij[1][:]+2*ncol])
            #Full_data  = np.hstack((Stiffpsv_data,Stiff_data))            
            #Full_ij    = np.hstack((Stiffpsv_ij,Stiffsh_ij))
            #Full_csc   = csc_matrix((Full_data,Full_ij)) 
            
            
            #Checking the Full matrix's shape in a picture            
            #AFull_csc= Full_csc.todense()          
            #scipy.misc.imsave('FullMatrix.png',1*(AFull_csc != 0.+0.j))              
               

        # Superlu
            lu = linalg.splu(Stiff_csc)
            ncol = Stiff.shape[1]
            g = np.zeros((ncol),complex)
            A = Stiff[:,i_0:]
            if abs(m) == 1:
                g[i_0:-1] =  -dW*(A[0][:-1]+A[1][:-1]+A[0][1:])
                g[-1] = -dW*(A[1][-1]+A[0][-1])
            else:
                g[i_0-1] = -dTw


            
            x = lu.solve(g)
            if abs(m) == 1: x[i_0:] += dW           
                        
            
            lupsv =  linalg.splu(Stiffpsv_csc)
            gpsv =  np.zeros((2*ncol),complex)
            Dis =   np.zeros((2*(ncol-i_0)),complex)
            ###Discontinuities in the displecements

            Ind =(Stiffpsv_ij[0,:] >= 2*i_0) * (Stiffpsv_ij[1,:] >= 2*i_0)
            Ind = np.nonzero(Ind)[0]
            Stiffsvp_up = csc_matrix((Stiffpsv_data[Ind],Stiffpsv_ij[:,Ind]-2*i_0)) 
            St_psv_Unp = Stiffsvp_up.todense()  
            #print St_psv_Unp[-2,-1]
            #Checking with a image
            #AStiffpsvp = Stiffpsvp.todense()   
            #scipy.misc.imsave('psvmat.png', 1*(AStiffpsvp != 0.))  
            Dis[::2]  += -dU 
            Dis[1::2] += -dV
            
            
            AUX1 = Stiff1[:,i_0:]
            AUX2 = Stiff2[:,i_0:]
            AUX3 = Stiff3[:,i_0:]
            AUX4 = Stiff4[:,i_0:]               
            
           
            if m==0:
                pass
            if abs(m)==1: 
		    
                # Uncomment to solve without use the unpack the matrix
                #~ gpsv[2*i_0+2:-3:2]=-dV*(AUX3[0,1:-1]+AUX4[1,1:-1]+AUX4[0,2:])
		#~ gpsv[2*i_0] = -dV*(AUX4[1,0]+AUX4[0,1])
                #~ gpsv[-2]=-dV*(AUX3[0,-1]+AUX4[1,-1])
                
                #~ gpsv[2*i_0+3:-2:2] = -dV*(AUX2[0,1:-1]+AUX2[1,1:-1]+AUX2[0,2:])
                #~ gpsv[2*i_0+1] = -dV*(AUX2[1,0]+AUX2[0,1])
                #~ gpsv[-1] = -dV*(AUX2[0,-1]+AUX2[1,-1])
#~ #
#
	         gpsv[2*i_0::] = np.dot(St_psv_Unp, Dis) 
                 #print np.shape(gpsv[2*i_0+1::2] ), np.shape(np.dot(St_psv_Unp, Dis)[0,1::2] )
            #np.savetxt('gpsv.txt', gpsv[2*i_0::].view(float).reshape(-1, 2))
	   ####  Excitations coeff       
            if m == 0:
                gpsv[2*i_0-2] = -dSv
                gpsv[2*i_0-1] = -dRu
            if abs(m) == 2:        
                gpsv[2*i_0-1] = -dSv

            xpsv = lupsv.solve(gpsv)   
            
            
            if abs(m) == 1: xpsv[2*i_0+1::2] += dV
            if m == 0: xpsv[2*i_0::2] += dU
            
        
            #print gpsv[2*i_0-1:] 
            #print g[i_0-1:]    
            #Compute the relative error at surface:
            erroru  = np.absolute(U0[-1]-xpsv[-2])/np.absolute(U0[-1]) * 100.
            errorv  = np.absolute(V0[-1]-xpsv[-1])/np.absolute(V0[-1]) * 100.
            errorw  = np.absolute(W0[-1]-x[-1])/np.absolute(W0[-1]) * 100.
            ERRORu[len(ERRORu):] = [erroru]
            ERRORv[len(ERRORv):] = [errorv]
            ERRORw[len(ERRORw):] = [errorw]

        Label = 'l=' + str(l)
#        pl.figure(1)
#        pl.plot(PERIOD,ERRORu,label=Label)
#        pl.legend()
#        
#        pl.figure(2)
#        pl.plot(PERIOD,ERRORv,label=Label)
#        pl.legend()
#        
#        pl.figure(3)
#        pl.plot(PERIOD,ERRORw,label=Label)
#        pl.legend()
        
        
        
        pl.figure()
        pl.plot(r,W0.real, label= "THEO")
        pl.plot(r,W0.imag+1.e-9, label= "THEO")
        pl.plot(rg[1:],x.real,label= "DSM")
        pl.plot(rg[1:],x.imag+1.e-9,label= "DSM")  
        pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,\
                  ncol=2, mode="expand", borderaxespad=0.)
        pl.title(Label+ "W")
        
        
        pl.figure()
        pl.plot(r,U0.real, label= "THEO")
        pl.plot(r,U0.imag+1.e-9, label= "THEO")
        pl.plot(rg[1:],xpsv.real[::2],label= "DSM")
        pl.plot(rg[1:],xpsv.imag[::2]+1.e-9,label= "DSM")
        pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,\
                  ncol=2, mode="expand", borderaxespad=0.)
        pl.title(Label+ "U")
        
                
        
        pl.figure()
        pl.plot(r,V0.real, label= "THEO")
        pl.plot(r,V0.imag+1.e-9, label= "THEO")
        pl.plot(rg[1:],xpsv.real[1::2],label= "DSM")
        pl.plot(rg[1:],xpsv.imag[1::2]+1.e-9,label= "DSM")
        pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,\
                  ncol=2, mode="expand", borderaxespad=0.)
        pl.title(Label+ "V")
    print ERRORu, ERRORv, ERRORw

    pl.show()

def gS1(r): return -1./r

def gS2(r): return 1.     

def gS3(r): return r*((LL*LL-1)/(r**2)-(omega**2)/2./beta_dsm**2)   

def gS4(r): return  -1.   

def gR1(r): return  2*mu_dsm*LL*LL/r/r-(lbd_dsm+2*mu_dsm)*omega**2/alpha_dsm**2    

def gR2(r): return  -4*mu_dsm/r    

def gR3(r): return -2*mu_dsm*LL*LL/r**2
        
def gR4(r): return  2*mu_dsm*LL*LL/r

   

if __name__=="__main__":
   main()

