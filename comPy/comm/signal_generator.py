-------------------
this file contains signal generators

Generate uniform-distributed random 0,1
Generate Bernoulli-distributed random binary numbers
Generate Poisson-distributed random integers
Generate integers randomly distributed in range [0, M-1]
Generate Barker Code
Generate Gold sequence from set of sequences
Generate Hadamard code from orthogonal set of codes
Generate Kasami sequence from set of Kasami sequences
Generate orthogonal variable spreading factor (OVSF) code from set of orthogonal codes
Generate pseudonoise sequence
Generate Walsh code from orthogonal set of codes
Read baseband signals from file

---------------------------------
import numpy as np 

def uniform(n):
	bits=np.random.randint(2,size=n)
	return bits

#generate uniform distribution ranodm 0,1

#