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
from compy.bin2gray import *
from compy.noise import *
from compy.modulation import*
__all__=['mimo_rayleigh_flat_fading','mimo_rician_flat_fading']

def mimo_rayleigh_flat_fading(nt,nr,norm='None'):
	h_re=np.random.randn(nr*nt)
	h_im=np.random.randn(nr*nt)
	h=h_re+1j*h_im
	if norm=='None':
		h=h
	elif norm=='unit_average':
		h=h/np.mean(abs(h))
	elif norm=='unit_peak':
		h=h/np.max(abs(h))
	else:
		print 'error norm type: only "unite average","unit peak"'
	reshape_h=np.reshape(h,(nr,nt))
	return reshape_h
	
def mimo_rician_flat_fading(nt,nr,mean1,mean2,sigma1,sigma2,norm='None'):
	h_re=mean1+sigma1*np.random.randn(nr*nt)
	h_im=mean2+sigma2*np.random.randn(nr*nt)
	h=h_re+1j*h_im

	if norm=='None':
		h=h
	elif norm=='unit_average':
		h=h/np.mean(abs(h))
	elif norm=='unit_peak':
		h=h/np.max(abs(h))
	else:
		print 'error norm type: only "unite average","unit peak"'
	reshape_h=np.reshape(h,(nr,nt))
	return reshape_h


def mimo_rayleigh_frequency_selective_fading():
	pass

def mimo_rician_frequency_selective_fading():
	pass


def siso_rayleigh_frequency_selective_fading():
	pass
