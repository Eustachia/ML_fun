# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:51:23 2017

@author: eust_abbondanza
"""

import random
import numpy as np
from knn import my_euclidean
import matplotlib.pyplot as plt

# randomly initiate datapoinst as cluster centroids

def generate_random_points(n_dim, n_point):
    data=[]
    for i in range(0, n_point):
      #  print (i)
        data.append([random.random() for _ in range(n_dim)])

    return np.asarray(data)

def initiate_centroids(data, n_cluster):
    centroids_ind=[]
    for n in range(0, n_cluster):
        centroids_ind.append(random.randint(0, len(data)))
    
    return data[centroids_ind]
 
#calculate distances from each datapoint to each centroids and assign dp to clusters

def assign_clusters(data, centroids):
   # print ("example centroids:", centroids[0:2])
    clusters=[]
    
    for i in range(0, len(data)):
        distances=[]
        for k in range(0, len(centroids)):
         
            distances.append(my_euclidean(data[i], centroids[k]))
 
        clusters.append((data[i], np.argmin(distances)))
    
 
    return clusters
 
def calculate_centroids(clusters, n_clust):
    centroids_upd=[0]*n_clust 
    labels=np.asarray([label for _, label in clusters])
 
    for k in range(0, n_clust):
   
        ind=np.array([labels==k])[0].flatten()
     
        clusters_tmp=np.asarray(clusters)[ind]
   
        centroids_upd[k]=  np.mean(clusters_tmp, axis=0)
        
    return [c for c, _ in centroids_upd]
    
    
if __name__ == "__main__":
    n_it=1000
    ndim=2
    nclust=4
    cost=np.zeros(n_it)
    data=generate_random_points(ndim, 1000)
    centroids=initiate_centroids(data, nclust)
    myclust=assign_clusters(data, centroids)
    
    clusters=[[] for i in range(nclust)]
  
    for item, clustNum in myclust:
        clusters[clustNum].append( item )
    counter=0  
    plt.figure(1)
    for cluster in clusters:
        mycolor=np.random.rand(nclust)
        plt.scatter([item[0] for item in cluster],[item[1] for item in cluster],color= mycolor)
        plt.scatter(centroids[counter][0], centroids[counter][1], color=mycolor, s=np.pi*100)
        counter=counter+1
    plt.show()
    
    for i in range(0, n_it):
     
        new_centroids=calculate_centroids(myclust, nclust)
      
       # print ("example new centroids", new_centroids[0:5])
   
        myclust=assign_clusters(data, new_centroids)
        if i % 10 == 0:
            print ("iteration", i)
    
        cost[i]=sum(np.power([my_euclidean(cl, np.asarray(new_centroids[lab][0])) for cl, lab in myclust], 2))/i
    plt.figure(2)
    plt.plot(cost)
    
    clusters=[[] for i in range(nclust)]
  
    for item, clustNum in myclust:
        clusters[clustNum].append( item )
    counter=0  
    plt.figure(3)
    for cluster in clusters:
        mycolor=np.random.rand(nclust)
        plt.scatter([item[0] for item in cluster],[item[1] for item in cluster],color= mycolor)
        plt.scatter(new_centroids[counter][0], new_centroids[counter][1], color=mycolor, s=np.pi*100)
        counter=counter+1
    plt.show()
 