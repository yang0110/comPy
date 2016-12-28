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
# m: order of modulation 
# input_bits_array: np.array of binary bits

__all__=['constellation','bpsk_mod','mpsk_mod','mpsk_ref_symbol','qam_ref_symbol','pam_ref_symbol','qam_mod','mpsk_dem',
'qam_dem','pam_mod','pam_dem','spatial_modulation_qam','sm_constellation','generalized_spatial_modulation_qam','gsm_ref_symbol_combination',
'gsm_look_up_table','mimo_look_up_table','ncr','Ber']


def constellation(data):
	re=np.real(data)
	im=np.imag(data)
	plt.scatter(re,im,s=50)
	plt.xlim(min(re)-1,max(re)+1)
	plt.ylim(min(im)-1,max(im)+1)
	plt.title('qma_%s'%(len(data)))
	plt.show()

def bpsk_mod(input_bits_array):
	bpsk=2*np.round(input_bits_array)-1
	return bpsk
	# output bits array [-1,1....]

def mpsk_mod(input_bits_array,m):
	# m_array=[2.0,4.0,8.0,16.0]
	m=float(m)
	input_ints=bits_to_binary_to_int(input_bits_array,m)
	I=np.cos(input_ints/m*2*np.pi+np.pi/4.0)
	Q=np.sin(input_ints/m*2*np.pi+np.pi/4.0)
	mpsk=I+1j*Q
	return mpsk

def mpsk_ref_symbol(m):
	m=float(m)
	ref_bits=np.arange(m)
	s_i=np.cos(ref_bits/m*2*np.pi+np.pi/4.0)
	s_q=np.sin(ref_bits/m*2*np.pi+np.pi/4.0)
	mpsk_ref_symbol=s_i+1j*s_q
	return mpsk_ref_symbol

def qam_ref_symbol(m):
	
	if m==8:
		m=16
		m=float(m)
		ref_values=np.arange(1,np.sqrt(m))
		ref_values=ref_values[0::2]
		v1=ref_values
		v2=ref_values*(-1)
		ref=np.hstack((v1,v2))
		ref_com=np.array(list(itertools.product(ref,repeat=2)))
		ref_symbol=ref_com[:,0]+1j*ref_com[:,1]
		qam=ref_symbol[np.where(abs(np.imag(ref_symbol))<=1)]
	elif m==32:
		m=64
		m=float(m)
		ref_values=np.arange(1,np.sqrt(m))
		ref_values=ref_values[0::2]
		v1=ref_values
		v2=ref_values*(-1)
		ref=np.hstack((v1,v2))
		ref_com=np.array(list(itertools.product(ref,repeat=2)))
		ref_symbol=ref_com[:,0]+1j*ref_com[:,1]
		qam=ref_symbol[np.where(abs(ref_symbol)<7.07)]
	else:
		m=float(m)
		ref_values=np.arange(1,np.sqrt(m))
		ref_values=ref_values[0::2]
		v1=ref_values
		v2=ref_values*(-1)
		ref=np.hstack((v1,v2))
		ref_com=np.array(list(itertools.product(ref,repeat=2)))
		ref_symbol=ref_com[:,0]+1j*ref_com[:,1]
		qam=ref_symbol
	return qam

def pam_ref_symbol(m,ini_phase):
	ref_symbol=np.arange(-(m-1),m,2)*np.exp(1j*ini_phase)
	return ref_symbol

def qam_mod(input_bits_array,m,type='binary'):
	#m_array=[4.0,16.0,64.0]
	m=float(m)
	ref_symbol=qam_ref_symbol(m)
	if type=='binary':
		input_ints=bits_to_binary_to_int(input_bits_array,m)
	elif type=='gray':
		input_ints=bits_to_gray_to_int(input_bits_array,m)
	else:
		print 'error type: type must be "binary" or "gray"' 
	input_sym=ref_symbol[input_ints]
	qam_symbol=input_sym
	return qam_symbol

def mpsk_dem(received_symbols,m):
	m=float(m)
	mpsk_symbol=mpsk_ref_symbol(m)
	mpsk_symbol=np.reshape(mpsk_symbol,(1,len(mpsk_symbol)))
	repeat_mpsk=np.repeat(mpsk_symbol,len(received_symbols),axis=0)
	reshape_received=np.reshape(received_symbols,(len(received_symbols),1))
	repeat_received=np.repeat(reshape_received,mpsk_symbol.shape[1],axis=1)
	distance=np.sqrt((np.real(repeat_received)-np.real(repeat_mpsk))**2+
						(np.imag(repeat_received)-np.imag(repeat_mpsk))**2)
	min_distance_index=np.argmin(distance,axis=1)
	return min_distance_index



def qam_dem(received_symbols,m):
	m=float(m)
	qam_symbol=qam_ref_symbol(m)
	qam_symbol=np.reshape(qam_symbol,(1,len(qam_symbol)))
	repeat_qam=np.repeat(qam_symbol,len(received_symbols),axis=0)
	reshape_received=np.reshape(received_symbols,(len(received_symbols),1))
	repeat_received=np.repeat(reshape_received,qam_symbol.shape[1],axis=1)
	distance=np.sqrt((np.real(repeat_received)-np.real(repeat_qam))**2+
						(np.imag(repeat_received)-np.imag(repeat_qam))**2)
	min_distance_index=np.argmin(distance,axis=1)
	return min_distance_index


def pam_mod(input_bits,m,ini_phase,type='binary'):
	m=float(m)

	if type=='binary':
		input_ints=bits_to_binary_to_int(input_bits,m)
	elif type=='gray':
		input_ints=bits_to_gray_to_int(input_bits,m)
	else:
		print 'error type: type must be "binary" or "gray"'
	ref_symbol=np.arange(-(m-1),m,2)*np.exp(1j*ini_phase)
	pam_symbol=ref_symbol[input_ints]
	return pam_symbol

def pam_dem(received_symbols,m,ini_phase):
	ref_symbol=np.arange(-(m-1),m,2)*np.exp(1j*ini_phase)
	ref_symbol=np.reshape(ref_symbol,(1,len(ref_symbol)))
	repeat_pam=np.repeat(ref_symbol,len(received_symbols),axis=0)
	reshape_received=np.reshape(received_symbols,(len(received_symbols),1))

	repeat_received=np.repeat(reshape_received,ref_symbol.shape[1],axis=1)
	distance=np.sqrt((np.real(repeat_received)-np.real(repeat_pam))**2+
						(np.imag(repeat_received)-np.imag(repeat_pam))**2)
	min_distance_index=np.argmin(distance,axis=1)
	received_ints=min_distance_index
	return received_ints


def spatial_modulation_qam(input_bits,nt,m,type='None'):
	k=np.log2(m)+np.log2(nt)
	a=np.log2(m)
	b=np.log2(nt)
	nb=len(input_bits)
	reshape_input_bits=np.transpose(np.reshape(input_bits,(nb/k,k)))
	symbol_input_bits=reshape_input_bits[:a,:]
	attenna_input_bits=reshape_input_bits[a:,:]

	symbol_input_bits2=np.reshape(np.transpose(symbol_input_bits),(1,
	symbol_input_bits.shape[0]*symbol_input_bits.shape[1])).ravel()
	attenna_input_bits2=np.reshape(np.transpose(attenna_input_bits),(1,
	attenna_input_bits.shape[0]*attenna_input_bits.shape[1])).ravel()
	if type=='None' or 'binary':

		symbol_input_int=bits_to_binary_to_int(symbol_input_bits2,m)
		attenna_input_int=bits_to_binary_to_int(attenna_input_bits2,nt)
	elif type=='gray':
		symbol_input_int=bits_to_gray_to_int(symbol_input_bits2,m)
		attenna_input_int=bits_to_gray_to_int(attenna_input_bits2,nt)

	else: 
		print 'error type: type must be "binary" or "gray"'
	norm_ref_symbol=qam_ref_symbol(m)
	norm_input_symbol=norm_ref_symbol[symbol_input_int]
	symbol_and_attenna=np.vstack((norm_input_symbol,attenna_input_int))

	X=np.zeros((nt,symbol_and_attenna.shape[1]))*(1j)
	for i in np.arange(symbol_and_attenna.shape[1]):
		attenna_number=int(symbol_and_attenna[1,i])
		X[attenna_number,i]=symbol_and_attenna[0,i]
		sm_modulated_symbol=X
	return sm_modulated_symbol

def sm_constellation(ref_symbol,nt):
	all_symbol_position=np.zeros((nt,nt*len(ref_symbol)))*1j
	for j in np.arange(len(ref_symbol)):
		for i in np.arange(j*nt,(j+1)*nt):
			all_symbol_position[i-j*nt,i]=ref_symbol[j]
	return all_symbol_position


def generalized_spatial_modulation_qam(input_bits,nt,n_act,m):
	nb_attenna_com=ncr(nt,n_act)
	a=np.log2(m)
	b=np.log2(2**np.floor(np.log2(nb_attenna_com)))
	nb=len(input_bits)
	k=float(a+b)
	reshape_input_bits=np.transpose(np.reshape(input_bits,((nb/k),k)))
	symbol_input_bits=reshape_input_bits[:a,:]
	attenna_input_bits=reshape_input_bits[a:,:]

	symbol_input_bits2=np.reshape(np.transpose(symbol_input_bits),(1,
	symbol_input_bits.shape[0]*symbol_input_bits.shape[1])).ravel()
	attenna_input_bits2=np.reshape(np.transpose(attenna_input_bits),(1,
	attenna_input_bits.shape[0]*attenna_input_bits.shape[1])).ravel()
	if type=='None' or 'binary':

		symbol_input_int=bits_to_binary_to_int(symbol_input_bits2,m)
		attenna_input_int=bits_to_binary_to_int(attenna_input_bits2,2**b)
	elif type=='gray':
		symbol_input_int=bits_to_gray_to_int(symbol_input_bits2,m)
		attenna_input_int=bits_to_gray_to_int(attenna_input_bits2,2**b)

	else: 
		print 'error type: type must be "binary" or "gray"'

	norm_ref_symbol=qam_ref_symbol(m)
	norm_input_symbol=norm_ref_symbol[symbol_input_int]
	symbol_and_attenna=np.vstack((norm_input_symbol,attenna_input_int))

	attenna_com=np.array(list(itertools.combinations(np.arange(nt),n_act)))
	nb_com=np.reshape(np.arange(len(attenna_com)),(len(attenna_com),1))
	nb_and_com=np.hstack((nb_com,attenna_com))
	

	X=np.zeros((nt,symbol_and_attenna.shape[1]))*(1j)
	for i in np.arange(symbol_and_attenna.shape[1]):
		attenna_number=(nb_and_com[symbol_and_attenna[1,i],1:]).astype(int)
		X[attenna_number,i]=symbol_and_attenna[0,i]
	return X


def gsm_ref_symbol_combination(nt,n_act,ref_symbol):
	attenna_combination=np.array(list(itertools.combinations(np.arange(nt),n_act)))
	b=2**np.floor(np.log2(len(attenna_combination)))
	attenna_combination=attenna_combination[:b,:]
	symbol_combination=np.reshape(ref_symbol,(len(ref_symbol),1))
	symbol_attenna_combination=np.array(list(itertools.product(symbol_combination,attenna_combination)))
	look_up_table1=np.transpose(symbol_attenna_combination)
	ref_symbol_combination=np.zeros((nt,look_up_table1.shape[1]))*1j
	for i in np.arange(look_up_table1.shape[1]):
		ref_symbol_combination[look_up_table1[1,i][0],i]=look_up_table1[0,i][0]
		ref_symbol_combination[look_up_table1[1,i][1],i]=look_up_table1[0,i][0]
	return ref_symbol_combination

def gsm_look_up_table(nt,n_act,ref_symbol):
	b=2**np.floor(np.log2(ncr(nt,n_act)))
	symbol_int_combination=np.arange(len(ref_symbol))
	symbol_attenna_int_combination=np.array(list(itertools.product(symbol_int_combination,np.arange(b))))
	return symbol_attenna_int_combination.astype(int)

def Ber(input_bits,cap_bits):
		ber=np.sum(cap_bits!=input_bits)/float(len(input_bits))
		return ber	

def ncr(n,r):
	import math
	f=math.factorial
	return f(n)/f(r)/f(n-r)

def mimo_look_up_table(nt,ref_symbol):
	symbol_order=np.reshape(np.arange(len(ref_symbol)),(1,len(ref_symbol)))
	row_1=np.repeat(symbol_order,4,axis=1)
	attenna_order=np.reshape(np.arange(nt),(1,nt))
	row_2=np.reshape(np.repeat(attenna_order,len(ref_symbol),axis=0),(1,nt*len(ref_symbol)))
	look_up_table=np.vstack((row_1,row_2))
	look_up_table=np.transpose(look_up_table)

	return look_up_table

# input_bits=np.random.randint(2,size=300)
# pam_modulation=pam_mod(input_bits,8,np.pi/4.0,'binary')
# constellation(pam_modulation)
# dem_pam=pam_dem(pam_modulation,8,np.pi/4.0)
# input_ints=bits_to_binary_to_int(input_bits,8)
# ber=np.sum(input_ints!=dem_pam)
# print ber

# input_bits=np.random.randint(2,size=300)
# pam_modulation=pam_mod(input_bits,8,np.pi/4.0,'gray')
# constellation(pam_modulation)
# dem_pam=pam_dem(pam_modulation,8,np.pi/4.0)
# input_ints=bits_to_gray_to_int(input_bits,8)
# ber=np.sum(input_ints!=dem_pam)
# print ber
