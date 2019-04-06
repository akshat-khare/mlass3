#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import math
import random
class Node:
    def __init__(self,numtheta):
        self.theta=[0.0]*numtheta
        self.downstream=None
        self.upstream=None
        self.output=None
        self.deljthetabydelthetaunit = [0.0]*numtheta
        self.deljthetabydelnet = None
def loadtheta(network,theta ):
    for i in range(len(network)):
        for j in range(len(network[i])):
            for k in range(len(network[i][j].theta)):
                network[i][j].theta[k] = theta[i][j][k]
    return network
def sigmoid(x):
    return (1.0/(1.0+math.exp(-x)))
def forwardpass(features, network):
    for i in range(len(network)):
        for j in range(len(network[i])):
            tempnode = network[i][j]
            temp=0.0
            if(i==0):
                #parse from input
                for k in range(len(tempnode.theta)):
                    if(k==len(tempnode.theta)-1):
                        temp += tempnode.theta[k] * 1.0
                    else:
                        temp += tempnode.theta[k] * features[k]
                
            else:
                #parse from last layer
                prevlayer = network[i-1]
                for k in range(len(tempnode.theta)):
                    if(k==len(tempnode.theta)-1):
                        temp += tempnode.theta[k] * 1.0
                    else:
                        temp += tempnode.theta[k] * prevlayer[k].output
            tempnode.output = sigmoid(temp)
    lastlayer = network[len(network)-1]
    templist=[]
    for i in range(len(lastlayer)):
        templist.append(lastlayer[i].output)
    return templist
def findcost(output, correctoutput):
    temp = 0.0
    for i in range(len(output)):
        temp += 0.5 * ((output[i]-correctoutput[i])**2)
    return temp
def findtotalcostarr(inparr,outarr,network):
    costarr=[]
    for i in range(len(inparr)):
        tempforwardpass= forwardpass(inparr[i],network)
        print("tempforwardpass is")
        print(tempforwardpass)
        tempcost = findcost(tempforwardpass,outarr[i])
        costarr.append(tempcost)
    avgcost=0.0
    for i in range(len(costarr)):
        avgcost += costarr[i]
    avgcost = avgcost/(len(costarr))
    return costarr, avgcost
def printdeljtheta(network):
    print("deljtheta is")
    templist=[]
    for i in range(len(network)):
        templist.append([])
        for j in range(len(network[i])):
            templist[i].append([])
            for k in range(len(network[i][j].theta)):
                templist[i][j].append(network[i][j].deljthetabydelthetaunit[k])
    print(templist)
    return
def neuralnet(inparr, outarr, hiddeninfo,learningrate, batchsize,costthres, sampletheta):
    numtrain = len(inparr)
    diminput= len(inparr[0])
    dimoutput = len(outarr[0])
    network=[]
    for i in range(len(hiddeninfo)):
        network.append([])
    network.append([])
    for i in range(len(hiddeninfo)):
        #i is layer iterator
        for j in range(hiddeninfo[i]):
            #j is sublayer num unit iterator
            if(i==0):
                tempnode = Node(diminput+1)
                network[i].append(tempnode)
            else:
                tempnode = Node(hiddeninfo[i-1]+1)
                network[i].append(tempnode)
            
    #add output layer
    for i in range(dimoutput):
        tempnode = Node(hiddeninfo[len(hiddeninfo)-1] + 1)
        network[len(network)-1].append(tempnode)
    #load theta for testing
#     network = loadtheta(network, sampletheta)
    #necessary forwardpass
#     tempforwardpass = forwardpass(inparr[0],network)
#     print(tempforwardpass)
#     #lets see cost
#     print("cost = "+str(findcost(tempforwardpass,outarr[0])))
    
    #time for backprop
    deljthetabydelthetamat=[]
    for i in range(len(network)):
        deljthetabydelthetamat.append([])
        for j in range(len(network[i])):
            deljthetabydelthetamat[i].append([])
            for k in range(len(network[i][j].theta)):
                deljthetabydelthetamat[i][j].append(0.0)
    print("simulated deljtheata")
    print(deljthetabydelthetamat)
    print("real deljtheta")
    printdeljtheta(network)
    subbatchiter=0
    batchcount=0
    numwholedatapass=0
    numiter=0
    costarr,oldavgcost = findtotalcostarr(inparr,outarr,network)
    shufflehelper = []
    for i in range(len(inparr)):
    	shufflehelper.append(i)
    while(0==0):
        numiter+=1
        print("numiter is")
        print(numiter)
        print("numwholedata pass is")
        print(numwholedatapass)
        ioindex = batchsize*batchcount+subbatchiter
        print("ioindex is "+str(ioindex))
        tempforwardpass = forwardpass(inparr[shufflehelper[ioindex]],network)
        for i in range(len(network)-1,-1,-1):
            #lets calculate deljtheta by del netj
            for j in range(len(network[i])):
                tempnode = network[i][j]
                if(i==len(network)-1):
                    tempnode.deljthetabydelnet = -1.0*(outarr[shufflehelper[ioindex]][j]-tempnode.output)*(tempnode.output)*(1.0-tempnode.output)
                else:
                    temp=0.0
                    nextlayer = network[i+1]
                    for k in range(len(nextlayer)):
                        temp += nextlayer[k].deljthetabydelnet * nextlayer[k].theta[j]
                    tempnode.deljthetabydelnet = (tempnode.output) * (1.0- tempnode.output) * temp
                if(i==0):
                    #ok is input
                    for k in range(len(tempnode.theta)):
                        if(k==len(tempnode.theta)-1):
                            tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * 1.0
                        else:
                            tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * inparr[shufflehelper[ioindex]][k]
                else:
                    prevlayer = network[i-1]
                    for k in range(len(tempnode.theta)):
                        if(k==len(tempnode.theta)-1):
                            tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * 1.0
                        else:
                            tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * prevlayer[k].output
        subbatchiter+=1
        for i in range(len(network)):
                for j in range(len(network[i])):
                    for k in range(len(network[i][j].theta)):
                        deljthetabydelthetamat[i][j][k] += network[i][j].deljthetabydelthetaunit[k]
        print("simulated deljtheata")
        print(deljthetabydelthetamat)
        print("real deljtheta")
        printdeljtheta(network)
        if(subbatchiter==batchsize):
            print("batch over")
            for i in range(len(network)):
                for j in range(len(network[i])):
                    for k in range(len(network[i][j].theta)):
                        network[i][j].theta[k] = network[i][j].theta[k] - learningrate * deljthetabydelthetamat[i][j][k]/(1.0*batchsize)
                        deljthetabydelthetamat[i][j][k]=0.0
            #set deljthetabydel zero
            
            
            
            subbatchiter=0
            batchcount+=1
            if(len(inparr)==batchcount*batchsize):
                print("shuffle the shufflehelper")
                numwholedatapass+=1
                batchcount=0
                random.shuffle(shufflehelper)

                costarr, newavgcost=findtotalcostarr(inparr,outarr,network)
                
                print("cost is")
                print(newavgcost)
                if(newavgcost<costthres):
                    print("cost is lower")
                    break
                elif(numwholedatapass>=20000):
                    break
                else:
                    #do nothing continue
                    oldavgcost=newavgcost
                
        
            
            
            
#         tempforwardpass = forwardpass(inparr[0],network)
#         print(tempforwardpass)
#         #lets see cost
#         print("cost = "+str(findcost(tempforwardpass,outarr[0])))        
    
    return network
# sampleinparr = [[0.05, 0.10]]
sampleinparr = [[0.0, 1.0],[1.0,0.0],[0.0,0.0],[1.0,1.0]]

# sampleoutput = [[0.01, 0.99]]
sampleoutput = [[0.99],[0.99],[0.01],[0.99]]
samplehiddeninfo = [1]
samplelearningrate=0.5
sampletheta = [[[.15,.20,.35],[.25,.30,.35]],[[.40,.45,.60],[.50,.55,.60]]]

samplenetwork = neuralnet(sampleinparr,sampleoutput,samplehiddeninfo,samplelearningrate,1, pow(10,-4),sampletheta)
print(samplenetwork)
            
                
    


# In[ ]:


pow(10,-3)

