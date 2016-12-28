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
import mpmath
from sympy.combinatorics.graycode import gray_to_bin

# m: order of modualtion
# binary bits: a matrix of binary bits. the # of rows is the length of each bin
# gray bits : a matrix of binary bits. the # of rows is the length of each bin

__all__=['binary_bits_to_gray_bits','gray_bits_to_binary_bits','bits_to_str','str_to_bits','gray_str_to_binary_str',
			'binary_str_to_int','bits_to_gray_to_int','bits_to_binary_to_int','int_to_gray_bits','int_to_binary_bits']

def binary_bits_to_gray_bits(binary_bits):
	binary_bits=np.transpose(binary_bits)
	rows,cols=binary_bits.shape
	gray_bits=np.zeros((rows,cols))
	for i in np.arange(rows):
		gray_bits[i,:]=np.hstack((binary_bits[i,0],
			(binary_bits[i,1:cols]^binary_bits[i,0:cols-1])))
	return np.transpose(gray_bits).astype(int)

def gray_bits_to_binary_bits(gray_bits):
	gray_bits=np.transpose(gray_bits)
	rows,cols=gray_bits.shape
	binary_bits=np.zeros((rows,cols)).astype(int)
	for i in np.arange(rows):
		binary_bits[i,0]=gray_bits[i,0]
		for j in np.arange(cols)[1:]:
			binary_bits[i,j]=binary_bits[i,j-1]^gray_bits[i,j]
	return np.transpose(binary_bits).astype(int)


def bits_to_str(bits,m):
	k=int(np.log2(m))
	string=[]
	for i in np.arange(bits.shape[1]):
		b=''
		for j in np.arange(k):
			b+=str(bits[j,i])
		string.append(b)
	return string


def str_to_bits(string,m):
	k=int(np.log2(m))
	bits=np.zeros((k,len(string)))
	for i in np.arange(len(string)):
		for j in np.arange(k):
			bits[j,i]=string[i][j]
	return bits.astype(int)


def gray_str_to_binary_str(gray_str):
	binary_str=[]
	for i in np.arange(len(gray_str)):
		binary_str.append(gray_to_bin(gray_str[i]))
	return binary_str


def binary_str_to_int(binary_str):
	ints=np.array([int(x,2) for x in binary_str])
	return ints


def bits_to_gray_to_int(input_bits,m):

	k=np.log2(m)
	reshape_bits=np.reshape(input_bits,(len(input_bits)/k,k))
	gray_bits=gray_bits_to_binary_bits(np.transpose(reshape_bits))
	gray_str=bits_to_str(gray_bits,m)
	integer=np.array([int(x,2) for x in gray_str])
	return integer


def bits_to_binary_to_int(input_bits,m):
	k=np.log2(m)
	reshape_bits=np.reshape(input_bits,(len(input_bits)/k,k))
	reshape_bits=np.transpose(reshape_bits)
	binary_str=bits_to_str(reshape_bits,m)
	integer=np.array([int(x,2) for x in binary_str])
	return integer


def int_to_gray_bits(input_int,m):
	k=int(np.log2(m))
	binary_str=np.array([format(x,'0%sb'%(k)) for x in input_int])
	binary_bits=str_to_bits(binary_str,m)
	gray_bits=binary_bits_to_gray_bits(binary_bits)
	return gray_bits


def int_to_binary_bits(input_int,m):
	k=int(np.log2(m))
	binary_str=np.array([format(x,'0%sb'%(k)) for x in input_int])
	binary_bits=str_to_bits(binary_str,m)
	return binary_bits

# ...test
# m=8
# k=np.log2(m)
# nb=k*10**3
# input_bits=np.random.randint(2,size=nb)
# integer=bits_to_gray_to_int(input_bits,m)
# integer2=bits_to_binary_to_int(input_bits,m)
# bits=str_to_bits(gray_str,m)
# input_int=np.array([1,2,3])
# b=int_to_gray_bits(input_int,8)
# c=int_to_binary_bits(input_int,8)