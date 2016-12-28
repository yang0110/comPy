import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import pylab
import itertools
from line_coding import polar_nrz
import math
from scipy import signal
from scipy.integrate import simps
import numpy.matlib

__all__=['awgn','awgn_complex_noise']



def awgn_complex_noise(l):
	n_re=np.random.randn(l)
	n_im=np.random.randn(l)
	n=n_re+1j*n_im
	noise=np.reshape(n,(l,1)).ravel()
	return noise


def awgn(l):
	noise=np.random.randn(l)
	return noise
