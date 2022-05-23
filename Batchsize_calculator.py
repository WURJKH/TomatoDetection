# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:07:01 2020

@author: Klein Holkenborg
"""


datasetsize = 40000

print()
print ("Appropirate batchsizes (dataset devisable to 0): ")
print()

for i in range(1, datasetsize):
    if datasetsize % i == 0:
        print (i)

print()
    
  