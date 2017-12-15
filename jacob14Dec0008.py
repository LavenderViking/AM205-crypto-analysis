import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import json
from bs4 import BeautifulSoup
import requests



#read data into pandas df
bitcoin = pd.read_csv('bitcoin_price.csv')
ethereum = pd.read_csv('ethereum_price.csv')
litecoin = pd.read_csv('litecoin_price.csv')
monero = pd.read_csv('monero_price.csv')
ripple = pd.read_csv('ripple_price.csv')
nem = pd.read_csv('nem_price.csv')
dash = pd.read_csv('dash_price.csv')

small_t = list(bitcoin["Date"]).index("Nov 01, 2015")
large_t = list(bitcoin["Date"]).index("Nov 01, 2014")

#From Nov 01, 2015 - Nov 01, 2017
bitcoin_s = bitcoin.iloc[0:small_t+1,:]
ethereum_s = ethereum.iloc[0:small_t+1,:]
litecoin_s = litecoin.iloc[0:small_t+1,:]
monero_s = monero.iloc[0:small_t+1,:]
ripple_s = ripple.iloc[0:small_t+1,:]
nem_s = nem.iloc[0:small_t+1,:]
dash_s = dash.iloc[0:small_t+1,:]

#From Nov 01, 2014 - Nov 01, 2017
bitcoin_l = bitcoin.iloc[0:large_t+1,:]
litecoin_l = litecoin.iloc[0:large_t+1,:]
monero_l = monero.iloc[0:large_t+1,:]
ripple_l = ripple.iloc[0:large_t+1,:]
dash_l = dash.iloc[0:large_t+1,:]

sp_ng_nasd = pd.read_csv('sp_ng_nasd.csv')
rv_sp_ng_nasd = sp_ng_nasd.iloc[::-1]
gold_df = pd.read_csv('LBMA-GOLD.csv')


gold = list(float(i) for i in gold_df.iloc[0:738,2])
sp500 = list(rv_sp_ng_nasd["SP500"])
gas = list(rv_sp_ng_nasd["Gas"])
nasdaq = list(rv_sp_ng_nasd["Nasdaq"])

gold_ = []
for i in range(len(gold)):
    if gold[i] != gold[i]:
        gold_.append(gold[i-1])
    else:
        gold_.append(gold[i])
gold_p = list(float(i) for i in gold_)

sp500_ = []
for i in range(len(sp500)):
    if sp500[i] == '.':
        sp500_.append(sp500[i-1])
    else:
        sp500_.append(sp500[i])
sp500_p = list(float(i) for i in sp500_)

gas_ = []
for i in range(len(gas)):
    if gas[i] == '.':
        gas_.append(gas[i-1])
    else:
        gas_.append(gas[i])
gas_p = list(float(i) for i in gas_)

nasdaq_ = []
for i in range(len(nasdaq)):
    if nasdaq[i] == '.':
        nasdaq_.append(nasdaq[i-1])
    else:
        nasdaq_.append(nasdaq[i])
nasdaq_p = list(float(i) for i in nasdaq_)

gold_c = pd.DataFrame(np.array([gold_p[i+1]-gold_p[i] for i in range(len(gold_p)-1)]))
sp500_c = pd.DataFrame(np.array([sp500_p[i+1]-sp500_p[i] for i in range(len(sp500_p)-1)]))
gas_c = pd.DataFrame(np.array([gas_p[i+1]-gas_p[i] for i in range(len(gas_p)-1)]))
nasdaq_c = pd.DataFrame(np.array([nasdaq_p[i+1]-nasdaq_p[i] for i in range(len(nasdaq_p)-1)]))

t1 = bitcoin_s["Close"]
bitcoin_c = pd.DataFrame(np.array([t1[i+1]-t1[i] for i in range(len(t1)-1)]))

t2 = ethereum_s["Close"]
ethereum_c = pd.DataFrame(np.array([t2[i+1]-t2[i] for i in range(len(t2)-1)]))

t3 = litecoin_s["Close"]
litecoin_c = pd.DataFrame(np.array([t3[i+1]-t3[i] for i in range(len(t3)-1)]))

t4 = monero_s["Close"]
monero_c = pd.DataFrame(np.array([t4[i+1]-t4[i] for i in range(len(t4)-1)]))

t5 = ripple_s["Close"]
ripple_c = pd.DataFrame(np.array([t5[i+1]-t5[i] for i in range(len(t5)-1)]))

t6 = nem_s["Close"]
nem_c = pd.DataFrame(np.array([t6[i+1]-t6[i] for i in range(len(t6)-1)]))

t7 = dash_s["Close"]
dash_c = pd.DataFrame(np.array([t7[i+1]-t7[i] for i in range(len(t7)-1)]))

small_df = pd.concat([bitcoin_c, ethereum_c,litecoin_c,monero_c,ripple_c,nem_c,dash_c], axis=1)
small_df.columns = ["bitcoin","ethereum","litecoin","monero","ripple","nem","dash"]

gold_c = pd.DataFrame(np.array([gold_p[i+1]-gold_p[i] for i in range(len(gold_p)-1)]))
sp500_c = pd.DataFrame(np.array([sp500_p[i+1]-sp500_p[i] for i in range(len(sp500_p)-1)]))
gas_c = pd.DataFrame(np.array([gas_p[i+1]-gas_p[i] for i in range(len(gas_p)-1)]))
nasdaq_c = pd.DataFrame(np.array([nasdaq_p[i+1]-nasdaq_p[i] for i in range(len(nasdaq_p)-1)]))

small_compare_df = pd.concat([bitcoin_c, ethereum_c,litecoin_c,monero_c,ripple_c,nem_c,dash_c,gold_c,sp500_c,gas_c,nasdaq_c], axis=1)
small_compare_df.columns = ["bitcoin","ethereum","litecoin","monero","ripple","nem","dash","gold","sp500","gas","nasdaq"]

from sklearn.preprocessing import StandardScaler
small_df_std = pd.DataFrame(StandardScaler().fit_transform(small_df))
small_df_std.columns = ["bitcoin","ethereum","litecoin","monero","ripple","nem","dash"]
small_compare_df_std = pd.DataFrame(StandardScaler().fit_transform(small_compare_df))
small_compare_df_std.columns = ["bitcoin","ethereum","litecoin","monero","ripple","nem","dash","gold","sp500","gas","nasdaq"]

cov = np.cov(small_df_std.values.T)     # covariance matrix
corr = np.corrcoef(small_df.values.T)  # correlation matrix
eig_vals, eig_vecs = np.linalg.eig(cov)

from numpy import array,identity,diagonal
from math import sqrt

def jacobiReal(a,tol = 1.0e-9): # Jacobi method

    def maxElem(a): # Find largest off-diag. element a[k,l]
        n = len(a)
        aMax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(a[i,j]) >= aMax:
                    aMax = abs(a[i,j])
                    k = i; l = j
        return aMax,k,l

    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: 
            t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0: 
                t = -t
        c = 1.0/sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l):  # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n):  # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n):      # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
        
    n = len(a)
    maxRot = 5*(n**2)       # Set limit on number of rotations
    p = identity(n)*1.0     # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop 
        aMax,k,l = maxElem(a)
        if aMax < tol: return diagonal(a),p
        rotate(a,p,k,l)
    print('Jacobi method did not converge')


def pp(a):
	def getMax(a):
        	maxVal = 0.0
		# Only need to check lower triangle of matrix:
       		for i in range(len(a)-1):
            		for j in range(i+1,len(a)):
                		if abs(a[i,j]) >= maxVal:
                    			maxVal,k,l = abs(a[i,j]),i,j
        	return maxVal,k,l

	biggest,ii,jj = getMax(a)

	for i in range(7):
		for j in range(7):
			if (ii==i and jj==j):
				print("X%.2fX" % (a[i,j]), end=' ')
			else:
				print("%.2f" % (a[i,j]), end=' ')
		print(" ")
	print("")

# Jacobi method
# Performs a series of plane rotations to calculate eigenvalues and
# eigenvectors of a matrix, a. Iteratively finds the largest element
# and makes it zero by performing a plane rotation on the matrix.
#
# Return:
# Returns eigenvalues for matrix a and the corresponding eigenvectors.
def jacobi(A,tol = 1.0e-9): # Jacobi method

    # Finds largest element and its indexes below the diagonal:
    def getMax(A):
        maxVal = 0.0
	# Only need to check lower triangle of matrix:
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                if abs(A[i,j]) >= maxVal:
                    maxVal,k,l = abs(A[i,j]),i,j
        return maxVal,k,l

    # Performs one plane rotation to eliminate the largest element
    # that is not on the diagonal.
    # Input:
    # a: square matrix that is initially the covariance matrix
    # p: the eigenvector matrix that is initally the identity matrix
    # 
    # Output (does not return the variables, changes by reference):
    # a: has the eigenvalues on the diagonal
    # p: stores the eigenvectors
    # rotCount: stores how many rotations were performed:
    def rotate(A,P,rotCount):

        # Performs a rotation at locations (i,j) and (k,l) in matrix M:
        # M : the matrix to perform rotation on
        # i,j: location of first element to rotate
        # k,l: location of second element to rotate
        def rotatePair(M,i,j,k,l):
            M_ij,M_kl = M[i,j],M[k,l]
        
            M[i,j] = M_ij - s*(M_kl+M_ij*tau)
            M[k,l] = M_kl + s*(M_ij-M_kl*tau)
        
        # Size of matrix:
        n = len(A)

        # Get indexes of max element:  
        maxVal,i,j = getMax(A)

        # Matrix has converged if the maximum non-diagonal value is less than epsilon
        # If we have performed 5*n^2 rotations and the matrix has not converged we return:
        if maxVal < tol or rotCount==5*n**2: return

        theta = (A[j,j]-A[i,i])/(2*A[i,j])
        t = np.sign(theta)/(abs(theta) + sqrt(theta**2 + 1.0))

        # Variables used for rotation:
        c = 1.0/sqrt(1.0 + t**2);
        s = t*c
        tau = s/(1.0 + c)

        # Rotation point and it's projection points on the diagonal:
        A_ij = A[i,j]

        # Case i)
        A[i,j] = 0.0
        
        # Case ii)
        A[i,i] = A[i,i]-t*A_ij
        A[j,j] = A[j,j]+t*A_ij

        # Case iii)
        # Rotation of elements:
        # Combines 3 cases of rotations (see picture TODO:):
        # i)   r < i
        # ii)  i < r < j
        # iii) j < r
        
        for r in range(i):
            rotatePair(A,r,i,r,j)
        for r in range(i+1,j):
            rotatePair(A,i,r,r,j)
        for r in range(j+1,n):
            rotatePair(A,i,r,j,r)
        for r in range(n):      # Update transformation matrix
            rotatePair(P,r,i,r,j)
        
        # Iterate until matrix converges:
        rotate(A,P,rotCount+1)

    # Initialize the eigenvector matrix:
    n,P,rotCount = len(A),identity(len(A))*1.0,0

    # Rotate until the matrix has converged:
    rotate(A,P,rotCount)

    if rotCount == 5*n**2:
        print('The Jacobi method did not converge')

    return diagonal(A),P


values,vectors = jacobi(cov,tol = 1.0e-9)
cov = np.cov(small_df_std.values.T)     # covariance matrix
values2,vectors2 = jacobiReal(cov,tol = 1.0e-9)

print(values)
print(vectors)




print(values-values2)
print(vectors-vectors2)



