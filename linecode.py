import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

__all__=['unipolar_nrz','unipolar_rz','polar_nrz','polar_rz',
		 'dipolar_ook','machester_nrz']
		 

	# input: 	 a stream of bits
	# rb:        the bit rate
	# amplitude: amplitude of output
	# fs: 		 sampling frequency
	# ts: 		 sampling period
	# tb: 		 bit period

def unipolar_nrz(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb 
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	unipolar_nrz_scheme=np.ones((1,sampling_ratio))
	output[np.where(input_bit_array==0)]=unipolar_nrz_scheme*0
	output[np.where(input_bit_array==0)]=unipolar_nrz_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio,1)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs




def unipolar_rz(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	unipolar_rz_scheme=np.hstack((np.ones((1,sampling_ratio/2)),np.zeros((1,sampling_ratio/2)))).ravel()
	output[np.where(input_bit_array==0)]=unipolar_rz_scheme*0
	output[np.where(input_bit_array==0)]=unipolar_rz_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio,1)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs


		

def polar_nrz(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	polar_nrz_scheme=np.ones((1,sampling_ratio)).ravel()
	output[np.where(input_bit_array==0)]=polar_nrz_scheme*(-1)*amplitude
	output[np.where(input_bit_array==1)]=polar_nrz_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio,1)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs


def polar_rz(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	polar_rz_scheme=np.hstack((np.ones((1,sampling_ratio/2)),np.zeros((1,sampling_ratio/2)))).ravel()
	output[np.where(input_bit_array==0)]=polar_rz_scheme*(-1)*amplitude
	output[np.where(input_bit_array==1)]=polar_rz_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio,1)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs


def dipolar_ook(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	dipolar_ook_scheme=np.hstack((np.ones((1,sampling_ratio/2)),np.ones((1,sampling_ratio/2))*(-1))).ravel()
	output[np.where(input_bit_array==0)]=dipolar_ook_scheme*0
	output[np.where(input_bit_array==1)]=dipolar_ook_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs


def manchester_nrz(input_bit_array,rb,sampling_ratio,amplitude):
	fs=sampling_ratio*rb
	ts=1./fs
	tb=1./rb
	output=np.zeros((len(input_bit_array),sampling_ratio))
	manchester_scheme=np.hstack((np.ones((1,sampling_ratio/2)),np.ones((1,sampling_ratio/2))*(-1))).ravel()
	output[np.where(input_bit_array==0)]=manchester_scheme*(-1)*amplitude
	output[np.where(input_bit_array==1)]=manchester_scheme*amplitude
	output=output.reshape((len(input_bit_array)*sampling_ratio,1)).ravel()
	time=np.arange(0,tb*len(input_bit_array),ts)
	return output,time,fs


# ..test 
# nb=100000
# input_bit_array=np.random.randint(2,size=nb)
# rb=4
# amplitude=5
# sampling_ratio=10
# output=unipolar_nrz(input_bit_array,4,10,5)
# print output.shape
# output=unipolar_rz(input_bit_array,4,10,5)
# output=polar_nrz(input_bit_array,4,10,5)
# output=polar_rz(input_bit_array,4,10,5)
# output=dipolar_ook(input_bit_array,4,10,5)
# output,time,fs=manchester_nrz(input_bit_array,4,10,5)
