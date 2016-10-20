# ----------------
# This file contains noise generators
# wgn

# -----------------------------------
import numpy as np 

from numpy import *

def wgn(mean,var,n,m=None):
	if m==None:
		m=1
		noise_array=np.random.normal(mean,var,n*m)
		noise_matrix=noise_array
	else:
		m=m
		noise_array=np.random.normal(mean,var,n*m)
		noise_matrix=np.reshape(noise_array,(n,m))
	return noise_matrix


# n*m samples are drawn from the same gaussian distribution,m default=1
# mean is the mean of the normal distribution 
# var is the standard deviation of the normal distribution


def wgn_multivariate(n,mean=None,cov=None):
	if mean==None:
		mean=[0,0]
	else:
		mean=mean
	if cov==None:
		cov=[[1,0],[0,1]]
	else:
		cov=cov

	noise_matrix=np.random.multivariate_normal(mean,cov,n)
	
	return noise_matrix

# default as no correlation between 2 variables from gaussion distribution
# n number of samples
# mean is a array of mean for m variables, the length is m 
# cov is the covariance matrix of size m by m for m variables,

def wgn_complex(n,mean=None,cov=None):

	if mean==None:
		mean=[0,0]
	else:
		mean=mean
	if cov==None:
		cov=[[1,0],[0,1]]
	else:
		cov=cov

	noise_matrix=np.random.multivariate_normal(mean,cov,n)
	noise_complex_matrix=noise_matrix[:,0]+1j*noise_matrix[:,1]

	return noise_complex_matrix

#default generate complex number x+j*y, x and y are drawn from 2 standard 
#normal distribution with no correlation
# generate complex number x+j*y, x and y are drawn from 2 normal distributions
# mean is a 1by 2 array, the mean of x and y
# cov is the covariance matrix of x and y


def wgn_normalized(array):
	array_absolute=np.absolute(array)
	normalized_array=array/array_absolute
	return normalized_array

# generate a complex array with each element has 1 absolute value 







