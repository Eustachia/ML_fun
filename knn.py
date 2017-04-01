# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:43:32 2017

@author: eust_abbondanza
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt

#generate data

def generate_random_points(n_dim, n_point, n_labels):
    data=[]
    for i in range(0, n_point):
      #  print (i)
        data.append(([random.random() for _ in range(n_dim)], random.randint(1, n_labels)))

    print ("data before:", data[0:5])
    data_new=[]
    for it, cl in data:
      #  print ("item before:", it, "cluster:", cl)
        it_new =np.asarray(it) + np.array([-2, 2])*(cl/2) 
        data_new.append([it_new, cl])
       # print ("item after:", it, "cluster:", cl)
    print ("data after:", data_new[0:5])
    return np.asarray(data_new)

def generate_n_cluster_data(n_dim, n_point, n_labels):
    X=[[] for i in range(n_labels)]
    Y=[0]*n_labels
    for cl in range(0, n_labels):
        mycl=np.array([random.randint(0, 6), random.randint(0, 6)])
        for p in range(0, n_point):
            X[cl].append([random.random() for _ in range(n_dim)+mycl])
        
   

        Y[cl] = np.array([cl]*n_point)
    
    return X, Y
# introduce a distance metrics - euclidean distance
def my_euclidean(a, b):
    tmp = a - b
    sum_squared = np.dot(tmp.T , tmp)
    return math.sqrt(sum_squared)

#calculate for each dp distances between other dps and choose n nearest

def find_nearest_neighbours(newpoint, data, n_neighbours):
    distances=[]
    for point in data:
      #  print (point[0])
      #  print (newpoint)
        distances.append((my_euclidean(np.asarray(newpoint), np.asarray(point[0])), point[1]))
    
   # distances = sorted(data, key=lambda datapoint: my_euclidean(datapoint, point))
    distances.sort()
   # distances.reverse()
    distances=distances[:n_neighbours]
 #   print (distances)
    nearest_neighbours = [label for _, label in distances]
  #  print (nearest_neighbours)
    return majority_vote(nearest_neighbours)

#get majority vote of their labels, return

def majority_vote(labeled_neighbours):
    
    numbers=[]
    labels=np.unique(labeled_neighbours).tolist()
   # label_list=[label for _, label in labeled_neighbours]
    #([label for _, label in labeled_neighbours])
    for v in labels:
        numbers.append((labeled_neighbours.count(v), v))
 #   print (numbers)
    numbers.sort()
    numbers.reverse()
    return numbers[0][1]

if __name__ == "__main__":
    n_dim=2
    n_points = 1000
    nclust=8
    n_neighb=7
    n_new_points=10
    mydata=generate_random_points(n_dim, n_points, nclust)
    print (mydata[0:5])
   # mydata, clusters=generate_n_cluster_data(n_dim, n_points, nclust)
  #  print ("data:", len(mydata), "labels:", len(clusters))
    
   # mydata=np.hstack([mydata, mylabels])    
    clusters=[[] for i in range(nclust)]
#  
    for item, clustNum in mydata:
      #  print (clustNum)
        clusters[clustNum-1].append( item )
    counter=0 
    mycolor=[]
    plt.figure(1)
    for cluster in clusters:
        mycolor.append(np.random.rand(4))
        plt.scatter([item[0] for item in cluster],[item[1] for item in cluster],color= mycolor[counter])
        
        counter=counter+1
    plt.show()
  #  print (mydata[0:5])
    for i in range(0, n_new_points):
        new_point=[random.random() for _ in range(n_dim)]
        new_point=new_point+np.array([-1*random.randint(1, nclust), random.randint(1, nclust)])
        label=find_nearest_neighbours(new_point, mydata, n_neighb)
        plt.scatter(new_point[0], new_point[1], color=mycolor[label-1], s=np.pi*100)
    #my_labeled_data=my_kmeans(mydata)
    
    print ("predicted label = ", label)