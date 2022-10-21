import numpy as np
from scipy import optimize as optimise

def gaussian(xs,a,mu,sigma):
    '''
    Defines a Gaussian curve
    
    INPUT
    -----
    xs: NumPy array
        The range of values over which the Gaussian is to be calculated
    a: flaot
        The scale factor for the Gaussian
    mu: float
        The mean of the Gaussian
    sigma: float
        The standard deviation of the Gaussian
    
    OUTPUT
    ------
    ys: NumPy array
        The y-values of the Gaussian, corresponding to the input x-values
    '''

    ys = a/np.sqrt(2*np.pi)*np.exp(-(xs-mu)**2/(2*sigma**2)) #Equation for a Gaussian
    #Note that because NumPy can do arithmatic on arrays as though they were single
    #numbers, we only need to put one line in here, and the output will be an array
    #of the same dimension as the input.

    return ys

def makeGauss(np_hist):
    '''
    Takes the input histogram from NumPy and returns a symmetric Gaussian \
    based off the negative side of the distribution.
    
    INPUT
    -----
    np_hist: tuple of arrays
        This should be the direct output of np.histogram, a tuple of two arrays. 
        The first array gives the heights of the bins, the second the bin edges.
    
    OUTPUT
    ------
    symmetric_xs: NumPy array
        The x-values of a symmetric Gaussian, matching the lower half of the input
    
    symmetric_ys: NumPy array
        The y-values of a symmetric Gaussian, matching the lower half of the input
    '''
    xs = []
    for i in range(len(np_hist[1])-1): #Iterate through the bin edges, up to the penultimate one
        xs.append(np.mean(np_hist[1][i:i+2])) #Add the midpoint of the bin to the array
    xs = np.array(xs)

    ys = np_hist[0] #xs and ys are now the midpoints and heights of the bins, respectively

    peak = np.max(ys)
    mean = xs[np.where(ys == peak)] #Define a mean of the new distribution as the peak of the input

    lower_ys = ys[np.where(xs < mean)] #Lower half of the Gaussian
    lower_xs = xs[np.where(xs < mean)]
    
    upper_ys = lower_ys[::-1] #Upper Gaussian
    upper_xs = xs[np.where(xs > mean)]
    upper_xs = upper_xs[:len(upper_ys)]

    symmetric_ys = np.concatenate([lower_ys,peak,upper_ys],axis=None) #Cat all the values into one array
    symmetric_ys = symmetric_ys.astype(np.float)
    symmetric_xs = np.concatenate([lower_xs,mean,upper_xs],axis=None)

    return symmetric_xs,symmetric_ys

def fitGauss(np_hist):
    '''
    Takes the input histogram from NumPy and returns a symmetric Gaussian \
    based off the negative side of the distribution.
    
    INPUT
    -----
    np_hist: tuple of arrays
        This should be the direct output of np.histogram, a tuple of two arrays. 
        The first array gives the heights of the bins, the second the bin edges.
    
    OUTPUT
    ------
    fit: tuple
        The scale, mean, and standard deviation of the best-fit Gaussian
    '''
    symmetric_xs,symmetric_ys = makeGauss(np_hist) #Get x and y for the Gaussian
    
    fit = optimise.curve_fit(gaussian,symmetric_xs,symmetric_ys,p0=(np.max(symmetric_ys),0.0,1.0)) #Fit parameters to the Gaussian

    return fit[0][0],fit[0][1],np.abs(fit[0][2])
    
