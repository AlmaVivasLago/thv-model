import numpy as np 

def denoise_xray(image_xray, delta_t=5e-3, time_steps=250, lmbda=0.01, norm='L1'):
    
    delta_t = delta_t
    time_steps = time_steps
    lmbda =  lmbda
    
    return _gradient_descent(image_xray.astype('float64'), time_steps, lmbda,  delta_t, norm)

def _l2_step(u, f, lmbda):
    """Computes L2 by finite differences. Assuptions 2 consecutive pixels NablaX = 1, Nablay = 1"""
    u_xx = np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)
    u_yy = np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)
    # Possible λ optimization 
    # λ = (-270.5*np.std(u) + 21572)/(np.std(u)**3 - 52.07*np.std(u)**2 + 1063*np.std(u) + 9677)
    u_t = -(u - f  - lmbda*(u_xx + u_yy))
    return  u_t

def _tv_step(u, f, lmbda):
    """Computes TV """
    u_x = (np.roll(u,-1,axis=1) -  np.roll(u,1,axis=1))/2
    u_y = (np.roll(u,-1,axis=0) -  np.roll(u,1,axis=0))/2
    u_xx = np.roll(u,-1,axis=1) - 2*u + np.roll(u,1,axis=1)
    u_yy = np.roll(u,-1,axis=0) - 2*u + np.roll(u,1,axis=0)
    u_xy = (np.roll(u_x,-1,axis=0) - np.roll(u_x,1,axis=0))/2
    # There is an impact on the solution depending on the choice of eps
    eps = 1e-7
    # Divergence
    u_t = -lmbda*(u-f)+ (((u_xx *(u_y**2)) + (u_yy*(u_x**2)) - 2*(u_x*u_y*u_xy))
                         /(eps+((u_x**2 + u_y**2))**(3/2)))
    return u_t

def _gradient_descent(f, time_steps, lmbda, delta_t, func):
    """Compute gradient descent optimazation technique, i.e., 
        TV or L2 regularization solver routine.
    Parameters
    ----------
    f : ndarray
        Noisy  image.
    time_steps : maximum number of iterations
        Test image.
    lmbda : float
        Parameter λ tunning balance between removing the noise and preserving the signal content.  
    delta_t : float
        Time constant, convergence rate.
    func: function modelling the noise, wich can be
        'L2' for the Gaussian noise model
        'L1'  for the Laplace noise model
    Returns
    -------
    u_t : ndarray
        Denoised image normalized [0, 1].    
    """
    # If there are no significative differences between u_t+1 and u_t, stop the loop
    u_t = np.copy(f)
    for i in range(1, time_steps):
        if func == 'L2':
            func =  _l2_step
            
        elif func == 'L1':
            func =  _tv_step
            
        step = func(u_t, f, lmbda)
        
        u_t1 = np.copy(u_t)
        
        u_t += (delta_t*step)

        if (np.abs(u_t1 - u_t) < delta_t).all():
            break
            
    return (u_t - u_t.min())/(u_t.max() - u_t.min())  