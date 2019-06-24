#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:32:02 2018

@author: jorge
"""

# Adaptive kernel smoothing (similar to Park et al., 2013)

# coords -- are the sample coordinates (e.g obtained by MDS).
# dim -- is the dimension of samples (or the number of features).
# K -- is the number of clusters to use in the adaptive variance (e.g. 
#      obtained by the silhouette score).
# obs -- is a vector of test points where the density will be evaluated.

# density -- is the evaluated density, the same length as obs.

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal

def adap_ks(coords, dim, K, obs):
    n = len(coords)
    mykmeans = KMeans(n_clusters=K, n_init=20, random_state=4855).fit(coords)
    cluscen = mykmeans.cluster_centers_
    labeled = np.zeros([np.size(coords, 0), dim+1])
    labeled[:,:-1] = coords
    labeled[:,-1] = mykmeans.labels_
    ninK = [np.count_nonzero(labeled[:,-1]==i) for i in range(K)]
    ind = np.argsort(labeled[:,-1])
    labeled = labeled[ind,:]
    loc = np.zeros([K+1, 1], dtype=int)
    sig2 = np.zeros([K, dim])
    k = 0
    loc[0,0] = 0
    for i in range(n):
        if int(labeled[i,-1]) != k:
            loc[k+1,0] = i
            if ninK[k] > 1:
                sig2[k,:] = np.var(labeled[loc[k,0]:loc[k+1,0],:-1], axis=0, ddof=1)
            else:
                disclus = np.linalg.norm(labeled[loc[k,0],:-1]*np.ones(cluscen.shape)-cluscen, axis=1)
                indf = np.argsort(disclus)[1]
                sig2[k,:] = np.var(np.row_stack((labeled[loc[k,0],:-1],cluscen[indf])), axis=0, ddof=1)
            k = k + 1
            if k == K-1:
                loc[K,0] = n
                if ninK[k] > 1:
                    sig2[K-1,:] = np.var(labeled[loc[K-1,0]:loc[K,0],:-1], axis=0, ddof=1)
                else:
                    disclus = np.linalg.norm(labeled[loc[K-1,0],:-1]*np.ones(cluscen.shape)-cluscen, axis=1)
                    indf = np.argsort(disclus)[1]
                    sig2[K-1,:] = np.var(np.row_stack((labeled[loc[K-1,0],:-1],cluscen[indf])), axis=0, ddof=1)                    
                break
        
    #silv = (4/(dim+2))**(1/(dim+4))*n**(-1/(dim+4)) # Silverman's rule of thumb
    density = np.zeros(np.size(obs,0))
    #print(silv**2*sig2)
    for j in range(len(obs)):
        k = 0
        for i in range(n):
            if i == loc[k+1,0]:
                k = k + 1
            #bw = silv**2*sig2[k,:]
            bw = (4/(dim+2))**(1/(dim+4))*ninK[k]**(-1/(dim+4))*np.sqrt(sig2[k,:]) # Silverman's rule of thumb
            density[j] = density[j] + multivariate_normal.pdf(obs[j], labeled[i,:-1], bw**2)
            #(1/np.sqrt(np.prod(bw**2)))*
    return density/n