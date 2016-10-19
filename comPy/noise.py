# ----------------
# This file contains noise generators
# wgn

# -----------------------------------
import numpy as np 

def wgn(a,b,n):
	noise=np.random.normal(a,b,n)
	return noise
	

# a is the mean of the normal distribution 
# b is the standard deviation of the normal distribution
# n is the number of elements of noise