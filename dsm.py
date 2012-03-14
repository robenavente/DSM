import numpy as np


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
    
def imats(rg,i_src):
    r1 = rg[1:]
    r0 = rg[0:-1]
    dr = r1-r0
    tmp11 = np.zeros(len(dr)+1)
    tmp22 = np.zeros(len(dr)+1.)

    # I0: r^2*X*X
    tmp12       = dr*(dr*dr*0.05 + r0*r1/6.) * (r0 != 0.)
    tmp11[0:-1] = dr*(dr*(dr*0.2 - r1*0.5) +r1*r1/3.) * (r0 != 0.)
    tmp22[1:]   = dr*(dr*(dr*0.2 + r0*0.5) +r0*r0/3.)
    J0 = tmp11[i_src]
    I0 = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I1: X*X 
    tmp11[0:-1] =  dr/3. * (r0 != 0.)
    tmp12       =  dr/6. * (r0 != 0.)
    tmp22[1:]   =  dr/3.
    J1 = tmp11[i_src]
    I1 = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I2a: r*X^dot*X
    tmp11[0:-1] = -(r1 + 2.*r0)/6. * (r0 != 0.)
    tmp12       = -(r0 + 2.*r1)/6. * (r0 != 0.)
    tmp22[1:]   =  (r0 + 2.*r1)/6.
    J2a = tmp11[i_src]
    I2a = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I2b: r*X*X^dot
    tmp11[0:-1] = -(r1 + 2.*r0)/6. * (r0 != 0.)
    tmp12       =  (r1 + 2.*r0)/6. * (r0 != 0.)
    tmp22[1:]   =  (r0 + 2.*r1)/6.
    J2b = tmp11[i_src]
    I2b = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I3: r^2*X^dot*X^dot
    tmp11[0:-1] =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.) * (r0 != 0.)
    tmp12       = -(r1*r1 + r1*r0 + r0*r0)/(dr*3.) * (r0 != 0.)
    tmp22[1:]   =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.)
    J3 = tmp11[i_src]
    I3 = np.array([tmp12,(tmp11+tmp22)[1:]])

    return (I0,I1,I2a,I2b,I3)

 
