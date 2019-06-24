#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:32:02 2018

@author: jorge
"""

# Adaptive kernel smoothing (similar to Park et al., 2013)
# added bias control on boundaries by reflection.
# parameter axis should be normalized to (0,1).

# coords -- are the sample coordinates (e.g obtained by MDS).
# dim -- is the dimension of samples (or the number of features).
# K -- is the number of clusters to use in the adaptive variance (e.g. 
#      obtained by the silhouette score).
# obs -- is a vector of test points where the density will be evaluated.

# density -- is the evaluated density, the same length as obs.

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal
from var_per import var_per

def adap_ks_evid_per(coords, dim, K, obs):
    n = len(coords)
    sincp = np.sin(coords[:,-1])
    coscp = np.cos(coords[:,-1])
    comcp = np.column_stack([coscp, sincp])
    fac = np.average([np.std(sincp),np.std(coscp)])/np.max(np.std(coords[:,:-1], axis=0))
    coordsclus = np.column_stack([fac*coords[:,:-1],comcp])
    mykmeans = KMeans(n_clusters=K, n_init=20, random_state=4855).fit(coordsclus)
    labeled = np.zeros([np.size(coords, 0), dim+1])
    labeled[:,:-1] = coords
    labeled[:,-1] = mykmeans.labels_
    ninK = [np.count_nonzero(labeled[:,-1]==i) for i in range(K)]
    cluscen = mykmeans.cluster_centers_
    ind = np.argsort(labeled[:,-1])
    labeled = labeled[ind,:]
    coordsclus = coordsclus[ind,:]
    loc = np.zeros([K+1, 1], dtype=int)
    sig2 = np.zeros([K, dim])
    k = 0
    loc[0,0] = 0
    for i in range(n):
        if int(labeled[i,-1]) != k:
            loc[k+1,0] = i
            if ninK[k] > 1:
                sig2[k,:-1] = np.var(labeled[loc[k,0]:loc[k+1,0],:-2], axis=0, ddof=1)
                sig2[k,-1] = var_per(labeled[loc[k,0]:loc[k+1,0],-2])
            else:
                disclus = np.linalg.norm(coordsclus[loc[k,0],:]*np.ones(cluscen.shape)-cluscen, axis=1)
                #print('coordsclus = ', coordsclus[loc[k,0],:])
                #print('disclus = ', disclus)
                indf = np.argsort(disclus)[1]
                #print('indf =', indf)
                sig2[k,:-1] = np.var(np.row_stack((labeled[loc[k,0],:-2],cluscen[indf,:-2])), axis=0, ddof=1)
                cpcluscen = np.arctan2(cluscen[indf,-2],cluscen[indf,-1])
                #print('cluscen =', cluscen)
                if cpcluscen >= 0:
                    cpcluscen = cpcluscen
                else:
                    cpcluscen = 2*np.pi+cpcluscen
                sig2[k,-1] = var_per(np.row_stack((labeled[loc[k,0],-2],cpcluscen)))
            k = k + 1
            if k == K-1:
                loc[K,0] = n
                if ninK[K-1] > 1:
                    sig2[K-1,:-1] = np.var(labeled[loc[K-1,0]:loc[K,0],:-2], axis=0, ddof=1)
                    sig2[K-1,-1] = var_per(labeled[loc[K-1,0]:loc[K,0],-2])
                else:
                    disclus = np.linalg.norm(coordsclus[loc[K-1,0],:]*np.ones(cluscen.shape)-cluscen, axis=1)
                    indf = np.argsort(disclus)[1]
                    sig2[K-1,:-1] = np.var(np.row_stack((labeled[loc[K-1,0],:-2],cluscen[indf,:-2])), axis=0, ddof=1)
                    cpcluscen = np.arctan2(cluscen[indf,-2],cluscen[indf,-1])
                    if cpcluscen >= 0:
                        cpcluscen = cpcluscen
                    else:
                        cpcluscen = 2*np.pi+cpcluscen
                    sig2[K-1,-1] = var_per(np.row_stack((labeled[loc[K-1,0],-2],cpcluscen)))
                break
        
    #print('sig2 = ', sig2)
    # silv = (4/(dim+2))**(1/(dim+4))*n**(-1/(dim+4)) # Silverman's rule of thumb
    density = np.zeros(np.size(obs,0))
    evid = 0.0
    fmom = 0.0
    # compute evidence
    k = 0
    for i in range(n):
        if i == loc[k+1,0]:
            k = k + 1
        bw = (4/(dim+2))**(1/(dim+4))*ninK[k]**(-1/(dim+4))*np.sqrt(sig2[k,:]) # Silverman's rule of thumb
        evid = evid + multivariate_normal.pdf(obs[0,:-1], labeled[i,:-2], bw[:-1]**2)
        fmom = fmom + (np.exp(np.complex(0,labeled[i,-2])))*multivariate_normal.pdf(obs[0,:-1], labeled[i,:-2], bw[:-1]**2)
    # compute density in observations
    for j in range(len(obs)):
        k = 0
        for i in range(n):
            if i == loc[k+1,0]:
                k = k + 1
            bw = (4/(dim+2))**(1/(dim+4))*ninK[k]**(-1/(dim+4))*np.sqrt(sig2[k,:])
            inside = multivariate_normal.pdf(obs[j], labeled[i,:-1], bw**2)
            lbound = multivariate_normal.pdf(obs[j], np.concatenate((labeled[i,:-2],labeled[i,-2:-1]+2*np.pi)), bw**2)
            rbound = multivariate_normal.pdf(obs[j], np.concatenate((labeled[i,:-2],labeled[i,-2:-1]-2*np.pi)), bw**2)
            density[j] = density[j] + inside + lbound + rbound
    
    meanang = np.arctan2(np.imag(fmom/evid),np.real(fmom/evid))
    meanang = (meanang + 2 * np.pi) % (2 * np.pi)
    return density/n, evid/n, mykmeans.labels_, meanang
