import numpy as np
global LL, lbd_dsm, mu_dsm,alpha_dsm, beta_dsm, omega, lbd_dsm

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
#    J0 = tmp11[i_src]
    I0 = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I1: X*X 
    tmp11[0:-1] =  dr/3. * (r0 != 0.)
    tmp12       =  dr/6. * (r0 != 0.)
    tmp22[1:]   =  dr/3.
#    J1 = tmp11[i_src]
    I1 = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I2a: r*X^dot*X
    tmp11[0:-1] = -(r1 + 2.*r0)/6. * (r0 != 0.)
    tmp12       = -(r0 + 2.*r1)/6. * (r0 != 0.)
    tmp22[1:]   =  (r0 + 2.*r1)/6.
#    J2a = tmp11[i_src]
    I2a = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I2b: r*X*X^dot
    tmp11[0:-1] = -(r1 + 2.*r0)/6. * (r0 != 0.)
    tmp12       =  (r1 + 2.*r0)/6. * (r0 != 0.)
    tmp22[1:]   =  (r0 + 2.*r1)/6.
#    J2b = tmp11[i_src]
    I2b = np.array([tmp12,(tmp11+tmp22)[1:]])
    # I3: r^2*X^dot*X^dot
    tmp11[0:-1] =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.) * (r0 != 0.)
    tmp12       = -(r1*r1 + r1*r0 + r0*r0)/(dr*3.) * (r0 != 0.)
    tmp22[1:]   =  (r1*r1 + r1*r0 + r0*r0)/(dr*3.)
#    J3 = tmp11[i_src]
    I3 = np.array([tmp12,(tmp11+tmp22)[1:]])

    return (I0,I1,I2a,I2b,I3)
    
 
    

 
