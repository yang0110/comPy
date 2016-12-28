import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import pylab
import numpy.matlib
import itertools
from line_coding import polar_nrz
import math
from scipy import signal
from scipy.integrate import simps
import numpy.matlib
from scipy.stats import gaussian_kde
from obspy.signal.util import nextpow2
from scipy import stats
import mpmath

__all__=['rayleigh_flat_fading_channel','doppler_filter','rayleigh_fading_clarke','rician_fading']


def rayleigh_flat_fading_channel(l):
	ch_re=np.random.randn(l)
	ch_im=np.random.randn(l)
	channel_values=ch_re+1j*ch_im
	H=channel_values
	return H

def doppler_filter(fm,fs,m):
	f=np.zeros((m,1)).ravel()
	doppler_ratio=fm/fs
	km=doppler_ratio*m
	for i in np.arange(1,m+1):
		print i
		if i==1:
			f[i-1]=0
		elif i>=2 and i<=km:
			f[i]=np.sqrt(1/(2*np.sqrt(1-(i/(m*doppler_ratio)**2))))
		elif i==km+1:
			f[i-1]=np.sqrt(km/2*(np.pi/2-np.arctan((km-1)/np.sqrt(2*km-1))))
		elif i>=km+2 and i<=m-km+2:
			f[i-1]=0
		elif i==m*km:
			f[i-1]=np.sqrt(km/2*(np.pi/2-np.arctan((km-1)/np.sqrt(2*km-1))))
		else:
			f[i-1]=np.sqrt(1/(2*np.sqrt(1-((m-i)/(m*doppler_ratio)**2))))
	fre_response=f
	return fre_response

def rayleigh_fading_clarke(M,N,fd,ts):
	#M number of multipath channel
	#N number of samples to generate
	#fd maximun doppler frequency
	#ts=sampling period
	a=0.0
	b=2*np.pi
	alpha=np.random.uniform(0,2*np.pi,size=M)
	beta=np.random.uniform(0,2*np.pi,size=M)
	theta=np.random.uniform(0,2*np.pi,size=M)
	h_re=np.zeros((1,N)).ravel()
	h_im=np.zeros((1,N)).ravel()

	m=np.linspace(1,M,M)
	for n in np.linspace(1,N,N):
		print h_re
		print h_im
		x=np.cos(((2*m-1)*np.pi+theta)/(4*M))
		h_re[n-1]=1/np.sqrt(M)*np.sum(np.cos(2*np.pi*fd*x*n*ts+alpha))
		h_im[n-1]=1/np.sqrt(M)*np.sum(np.sin(2*np.pi*fd*x*n*ts+beta))
		h=h_re+1j*h_im
	return h

def rician_fading(l,mean1,mean2,sigma1,sigma2):
	h_re=mean1+sigma1*np.random.randn(l)
	h_im=mean2+sigma2*np.random.randn(l)
	channel_values=h_re+1j*h_im
	H=channel_values
	return H


