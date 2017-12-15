import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import json
from bs4 import BeautifulSoup
import requests
from copy import deepcopy

# Jacobi method
# Performs a series of plane rotations to calculate eigenvalues and
# eigenvectors of a matrix, A. Iteratively finds the largest element
# and makes it zero by performing a plane rotation on the rows and columns
# where the largest element is located.
#
# Return:
# Returns eigenvalues for matrix A and the corresponding eigenvectors.
def jacobi(A,tol = 1.0e-9): # Jacobi method
    
    # Sorts the eigenvalues and keeps the order of the corresponding
    # eigenvectors.
    def sortResults(values,vectors):
        vectors = vectors.T
        
        indexes = list(reversed(np.argsort(values)))
        values = sorted(values,reverse=True)
        
        temp = deepcopy(vectors)
        
        for i in range(len(vectors)):
            vectors[i][:] = temp[indexes[i]][:]
        
        return (values,vectors)
    
    # Finds largest element and its indexes above the diagonal:
    def getMax(A):
        maxVal = 0.0
        # Only need to check upper triangle of matrix:
        for i in range(len(A)-1):
            for j in range(i+1,len(A)):
                if abs(A[i,j]) >= maxVal:
                    maxVal,k,l = abs(A[i,j]),i,j
        return maxVal,k,l
    
    # Performs one plane rotation to eliminate the largest element
    # that is not on the diagonal.
    # Input:
    # A: square matrix that is initially the covariance matrix
    # P: the eigenvector matrix that is initally the identity matrix
    #
    # Output (does not return the variables; changes by reference):
    # A: has the eigenvalues on the diagonal
    # P: stores the eigenvectors
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
        if maxVal < tol or rotCount==5*n**2:
            return
        
        # Variables used for rotation:
        theta = (A[j,j]-A[i,i])/(2*A[i,j])
        t = np.sign(theta)/(abs(theta) + sqrt(theta**2 + 1.0))
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
        # Combines types of rotations (see picture in PDF):
        
        for r in range(i): # i)   r < i
            rotatePair(A,r,i,r,j)
        for r in range(i+1,j): # ii)  i < r < j
            rotatePair(A,i,r,r,j)
        for r in range(j+1,n): # iii) j < r
            rotatePair(A,i,r,j,r)
        
        # Update eigenvector matrix:
        for r in range(n):
            rotatePair(P,r,i,r,j)
        
        # Iterate until matrix converges:
        rotate(A,P,rotCount+1)
    
    # Initialize the eigenvector matrix:
    n,P,rotCount = len(A),identity(len(A))*1.0,0
    
    # Rotate until the matrix has converged:
    rotate(A,P,rotCount)
    
    if rotCount == 5*n**2:
        print('The Jacobi method did not converge')
    
    A,P = sortResults(diagonal(A),P)
    return A,P

cov = np.cov(small_df_std.values.T)     # covariance matrix

values,vectors = jacobi(cov,tol = 1.0e-9)




