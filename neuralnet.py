#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import math
import random
import sys
import time
#greatest priority print
debug0=1
#greater priority print
debug1=0
#less priority print
debug2=0
class Node:
    def __init__(self,numtheta):
        self.theta=[0.0]*numtheta
        for i in range(numtheta):
            self.theta[i]=random.random()
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
# def forwardpass(features, network):
#     for i in range(len(network)):
#         for j in range(len(network[i])):
#             tempnode = network[i][j]
#             temp=0.0
#             if(i==0):
#                 #parse from input
#                 for k in range(len(tempnode.theta)):
#                     if(k==len(tempnode.theta)-1):
#                         temp += tempnode.theta[k] * 1.0
#                     else:
#                         temp += tempnode.theta[k] * features[k]
                
#             else:
#                 #parse from last layer
#                 prevlayer = network[i-1]
#                 for k in range(len(tempnode.theta)):
#                     if(k==len(tempnode.theta)-1):
#                         temp += tempnode.theta[k] * 1.0
#                     else:
#                         temp += tempnode.theta[k] * prevlayer[k].output
#             tempnode.output = sigmoid(temp)
#     lastlayer = network[len(network)-1]
#     templist=[]
#     for i in range(len(lastlayer)):
#         templist.append(lastlayer[i].output)
#     return templist
def forwardpass(features, network, ojnparr, thetanparr):
    for i in range(len(network)):
        if(i==0):
            tempinp = np.array(features).reshape(len(features),1)
            tempinp = np.append(tempinp, [1.0]).reshape(len(features)+1,1)
            temp = np.matmul(thetanparr[i],tempinp)
            temp = -1.0*temp
            temp = np.exp(temp)
            temp = 1.0+temp
            temp = 1.0 / temp
            ojnparr[i]= np.copy(temp)
        else:
            temp = np.append(ojnparr[i-1],[1.0]).reshape(len(ojnparr[i-1])+1,1)
            temp = np.matmul(thetanparr[i],temp)
            temp = -1.0*temp
            temp = np.exp(temp)
            temp = 1.0+temp
            temp = 1.0 / temp
            ojnparr[i]= np.copy(temp)
    return ojnparr
def findcost(output, correctoutput):
    temp = 0.0
    for i in range(len(output)):
        temp += 0.5 * ((output[i]-correctoutput[i])**2)
    temp = 0.5* (np.power(output-correctoutput,2))    
    return np.sum(temp)
def findtotalcostarr(inparr,outarr,network,ojnparr,thetanparr):
    costarr=[]
    for i in range(len(inparr)):
        tempout = np.array(outarr[i]).reshape(len(outarr[i]),1)
        tempforwardpass= forwardpass(inparr[i],network, ojnparr, thetanparr)
        if(debug1==1): print("tempforwardpass is")
        if(debug1==1): print(tempforwardpass)
        tempcost = findcost(tempforwardpass[len(network)-1],tempout)
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
def neuralnet(inparr, outarr, hiddeninfo,learningrate, batchsize,costthres,maxepochallowed,streak,sampletheta):
    starttime = time.time()
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
#     deljthetabydelthetamat=[]
#     for i in range(len(network)):
#         deljthetabydelthetamat.append([])
#         for j in range(len(network[i])):
#             deljthetabydelthetamat[i].append([])
#             for k in range(len(network[i][j].theta)):
#                 deljthetabydelthetamat[i][j].append(0.0)
    deljthetabydelthetamatnparr = []
    deljthetabydelthetanparr = []
    deljthetabydelnetnparr = []
    thetanparr = []
    ojnparr = []
    for i in range(len(network)):
        tempmat = np.zeros((len(network[i]), len(network[i][0].theta)))
        deljthetabydelthetamatnparr.append(tempmat)
        tempmat1 = np.zeros((len(network[i]), len(network[i][0].theta)))
        deljthetabydelthetanparr.append(tempmat1)
        tempmat2 = np.zeros((len(network[i]),1))
        deljthetabydelnetnparr.append(tempmat2)
        tempmat3 = np.zeros((len(network),1))
        ojnparr.append(tempmat3)
        tempmat4 = np.zeros((len(network[i]), len(network[i][0].theta)))
        for j in range(len(network[i])):
            for k in range(len(network[i][j].theta)):
                tempmat4[j][k]=network[i][j].theta[k]
        thetanparr.append(tempmat4)
        
#     if(debug2==1): print("simulated deljtheata")
#     if(debug2==1): print(deljthetabydelthetamat)
    if(debug2==1): print("real deljtheta")
#     if(debug2==1): printdeljtheta(network)
    if(debug2==1): print(deljthetabydelthetamatnparr)
    subbatchiter=0
    batchcount=0
    numwholedatapass=0
    numiter=0
    costarr,oldavgcost = findtotalcostarr(inparr,outarr,network, ojnparr, thetanparr)
    shufflehelper = []
    for i in range(len(inparr)):
        shufflehelper.append(i)
    streakiter=0
    while(0==0):
        numiter+=1
        if(debug1==1): print("numiter is")
        if(debug1==1): print(numiter)
        if(debug1==1): print("numwholedata pass is")
        if(debug1==1): print(numwholedatapass)
        ioindex = batchsize*batchcount+subbatchiter
        if(debug1==1): print("ioindex is "+str(ioindex))
        tempforwardpass = forwardpass(inparr[shufflehelper[ioindex]],network,ojnparr,thetanparr)
        if(debug1==1): print("real index is "+str(shufflehelper[ioindex]))
#         for i in range(len(network)-1,-1,-1):
            #lets calculate deljtheta by del netj
#             for j in range(len(network[i])):
#                 tempnode = network[i][j]
#                 if(i==len(network)-1):
#                     tempnode.deljthetabydelnet = -1.0*(outarr[shufflehelper[ioindex]][j]-tempnode.output)*(tempnode.output)*(1.0-tempnode.output)
#                 else:
#                     temp=0.0
#                     nextlayer = network[i+1]
#                     for k in range(len(nextlayer)):
#                         temp += nextlayer[k].deljthetabydelnet * nextlayer[k].theta[j]
#                     tempnode.deljthetabydelnet = (tempnode.output) * (1.0- tempnode.output) * temp
#                 if(i==0):
#                     #ok is input
#                     for k in range(len(tempnode.theta)):
#                         if(k==len(tempnode.theta)-1):
#                             tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * 1.0
#                         else:
#                             tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * inparr[shufflehelper[ioindex]][k]
#                 else:
#                     prevlayer = network[i-1]
#                     for k in range(len(tempnode.theta)):
#                         if(k==len(tempnode.theta)-1):
#                             tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * 1.0
#                         else:
#                             tempnode.deljthetabydelthetaunit[k] = tempnode.deljthetabydelnet * prevlayer[k].output
        for i in range(len(network)-1,-1,-1):
            #lets calculate deljtheta by del netj
#            print("iterating on layer "+str(i))
 #           print("theta is ")
  #          print(thetanparr[i])
   #         print("deljthetabydelnp")
    #        print(deljthetabydelnetnparr[i])
            if(i==len(network)-1):
                tempnp = np.array(outarr[shufflehelper[ioindex]]).reshape(len(outarr[shufflehelper[ioindex]]),1) - ojnparr[i]
                tempnp = np.multiply(tempnp, ojnparr[i])
                tempnp = np.multiply(tempnp, 1 - ojnparr[i])
                tempnp = -1.0 * tempnp
                deljthetabydelnetnparr[i] = np.copy(tempnp)
            else:
                tempnp = np.matmul(np.transpose(deljthetabydelnetnparr[i+1]),thetanparr[i+1])
                tempnp = np.transpose(tempnp)
                #print("tempnp is")
                #print(tempnp)
                tempnp = np.delete(tempnp, len(tempnp)-1,0)
                #print("tempnp is")
                #print(tempnp)
                #print("ojnparr i is")
                #print(ojnparr[i])
                tempnp2 = 1.0 - ojnparr[i]
                #print("tempnp2 is ")
                #print(tempnp2)
                tempnp2 = np.multiply(ojnparr[i],tempnp2)
                #print("tempnp2 is ")
                #print(tempnp2)
                tempnp = np.multiply(tempnp2,tempnp)
                deljthetabydelnetnparr[i] = np.copy(tempnp)
            #lets calculate deljtheta by del theta
            if(i==0):
                tempnp = np.append(inparr[shufflehelper[ioindex]],[1.0]).reshape(len(inparr[shufflehelper[ioindex]])+1,1)
                deljthetabydelthetanparr[i]=np.copy(np.matmul(deljthetabydelnetnparr[i],np.transpose(tempnp)))
            else:
                tempnp = np.append(ojnparr[i-1],[1.0]).reshape(len(ojnparr[i-1])+1,1)
                deljthetabydelthetanparr[i]=np.copy(np.matmul(deljthetabydelnetnparr[i],np.transpose(tempnp)))
        subbatchiter+=1
#         for i in range(len(network)):
#                 for j in range(len(network[i])):
#                     for k in range(len(network[i][j].theta)):
#                         deljthetabydelthetamat[i][j][k] =deljthetabydelthetamat[i][j][k] + network[i][j].deljthetabydelthetaunit[k]
        for i in range(len(network)):
            deljthetabydelthetamatnparr[i] = deljthetabydelthetamatnparr[i]+ deljthetabydelthetanparr[i]
        if(debug2==1): print("simulated deljtheata mat")
        if(debug2==1): print(deljthetabydelthetamatnparr)
        if(debug2==1): print("real deljtheta")
        if(debug2==1): print(deljthetabydelthetamatnparr)
        if(subbatchiter==batchsize):
            if(debug1==1): print("batch over")
#             for i in range(len(network)):
#                 for j in range(len(network[i])):
#                     for k in range(len(network[i][j].theta)):
#                         network[i][j].theta[k] = network[i][j].theta[k] - learningrate * deljthetabydelthetamat[i][j][k]/(1.0*batchsize)
#                         deljthetabydelthetamat[i][j][k]=0.0
            for i in range(len(network)):
                thetanparr[i] = thetanparr[i] - (learningrate * deljthetabydelthetamatnparr[i])
                deljthetabydelthetamatnparr[i] = np.zeros((len(network[i]), len(network[i][0].theta)))
            #set deljthetabydel zero
            
            
            
            subbatchiter=0
            batchcount+=1
            if(len(inparr)==batchcount*batchsize):
                if(debug0==1): print("whole dataset done")

                if(debug1==1): print("shuffle the shufflehelper")
                numwholedatapass+=1
                if(debug0==1): print(numwholedatapass)
                batchcount=0
                if(batchsize!=1):
                    random.shuffle(shufflehelper)

                costarr, newavgcost=findtotalcostarr(inparr,outarr,network, ojnparr,thetanparr)
                
                if(debug0==1): print("cost is")
                if(debug0==1): print(newavgcost)
                if(abs(oldavgcost-newavgcost)<costthres and oldavgcost>newavgcost):
                    print("oldcost is "+str(oldavgcost))
                    print("newcost is "+str(newavgcost))
                    print("cost is lower")
                    break
                elif(newavgcost-oldavgcost>0.1 and newavgcost > oldavgcost):
                    print("oldcost is "+str(oldavgcost))
                    print("newcost is "+str(newavgcost))
                    print("cost is diverged")
                    break
                elif(newavgcost-oldavgcost>0.0001):
                    print("oldcost is "+str(oldavgcost))
                    print("newcost is "+str(newavgcost))
                    print("cost is lower")
                    streakiter+=1
                    if(streakiter>streak):
                        break
                    else:
                        oldavgcost=newavgcost
                elif(numwholedatapass>=maxepochallowed):
                    break
                else:
                    #do nothing continue
                    oldavgcost=newavgcost
                
        
            
            
            
#         tempforwardpass = forwardpass(inparr[0],network)
#         print(tempforwardpass)
#         #lets see cost
#         print("cost = "+str(findcost(tempforwardpass,outarr[0])))        
    endtime=time.time()
    print(endtime-starttime)
    return network,ojnparr,thetanparr
#sampleinparr = [[0.05, 0.10]]
# sampleinparr = [[0.0, 1.0],[1.0,0.0],[0.0,0.0],[1.0,1.0]]
sampleinparr = [[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]]
#sampleoutput = [[0.01, 0.99]]
# sampleoutput = [[0.99],[0.99],[0.01],[0.99]]
sampleoutput=[[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
samplehiddeninfo = [2]
samplelearningrate=0.1
sampletheta = [[[.15,.20,.35],[.25,.30,.35]],[[.40,.45,.60],[.50,.55,.60]]]

samplenetworksample,ojnparr, thetanparr = neuralnet(sampleinparr,sampleoutput,samplehiddeninfo,samplelearningrate,1, pow(10,-5),1000,3,sampletheta)
print(samplenetworksample)
            

    


# In[53]:


def tester(inparr,outarr,network,ojnparr,thetanparr):
    correct=0
    wrong=0
    for i in range(len(inparr)):
    # for i in range(numtrain):
        tempforwardpass= forwardpass(inparr[i],network,ojnparr,thetanparr)
    #     tempforwardpass= forwardpass(xarrtrain[i],samplenetwork)
        #find max prob class
        maxindex=0
        maxval=tempforwardpass[len(tempforwardpass)-1][0]
        for j in range(len(outarr[i])):
            if(tempforwardpass[len(tempforwardpass)-1][j]>maxval):
                maxindex=j
                maxval=tempforwardpass[len(tempforwardpass)-1][j]
        if(abs(outarr[i][maxindex]-1.0)<0.0001):
#             print("correct")
            correct+=1
        else:
#             print("wrong")
            wrong+=1
    return((1.0*correct)/(1.0*(correct+wrong)))
acc = tester(sampleinparr,sampleoutput,samplenetworksample,ojnparr,thetanparr)
print(acc)


# In[36]:


trainfname = open('ass3data/poker-hand-training-true.data', 'r')
xarrtrainori=[]
yarrtrainori=[]
numtrain=0
for line in trainfname:
    linearr=line.split(',')
    xarrtrainori.append([])
    yarrtrainori.append([])
    for i in range(len(linearr)-1):
        xarrtrainori[numtrain].append(int(linearr[i]))
    yarrtrainori[numtrain].append(int(linearr[len(linearr)-1]))
    numtrain+=1
xnumchildhelper = [4,13,4,13,4,13,4,13,4,13]
ynumchildhelper = [10]
print(len(xarrtrainori[0]))


# In[37]:


#one hot encoding
xnumfeatures=0
xarrtrain = []
for i in range(len(xarrtrainori[0])): 
    tempcat = xnumchildhelper[i]
    for j in range(tempcat):
        xarrtrain.append([])
    for j in range(numtrain):
        tempval = xarrtrainori[j][i]
        for k in range(tempcat):
            if(k==tempval):
                xarrtrain[xnumfeatures+k].append(1.0)
            else:
                xarrtrain[xnumfeatures+k].append(0.0)
    xnumfeatures += tempcat
print("done")
# print(len(xarrtrain))
# print(xnumfeatures)
xarrtrain = np.array(xarrtrain).reshape((xnumfeatures,numtrain))
xarrtrain = np.transpose(xarrtrain)
xarrtrain = xarrtrain.tolist()
# print(len(xarrtrain))
# print(len(xarrtrain[0]))

ynumfeatures=0
yarrtrain = []
for i in range(len(yarrtrainori[0])): 
    tempcat = ynumchildhelper[i]
    for j in range(tempcat):
        yarrtrain.append([])
    for j in range(numtrain):
        tempval = yarrtrainori[j][i]
        for k in range(tempcat):
            if(k==tempval):
                yarrtrain[ynumfeatures+k].append(1.0)
            else:
                yarrtrain[ynumfeatures+k].append(0.0)
    ynumfeatures += tempcat
print("done")
# print(len(yarrtrain))
# print(ynumfeatures)
yarrtrain = np.array(yarrtrain).reshape((ynumfeatures,numtrain))
yarrtrain = np.transpose(yarrtrain)
yarrtrain = yarrtrain.tolist()
# print(len(yarrtrain))
# print(len(yarrtrain[0]))


# In[ ]:





# In[44]:


#train model
hiddeninfoarr = [5]
trainlearningrate=0.1
trainbatchsize=1
trainthreshold=pow(10,-5)

#greatest priority print
debug0=1
#greater priority print
debug1=0
#less priority print
debug2=0
maxepochval=200
# finalnetwork100,finojnparr100,finthetanparr100  = neuralnet(xarrtrain[:100],yarrtrain[:100],hiddeninfoarr,trainlearningrate,trainbatchsize, trainthreshold,maxepochval,sampletheta)
finalnetwork,finojnparr,finthetanparr = neuralnet(xarrtrain,yarrtrain,hiddeninfoarr,trainlearningrate,trainbatchsize, trainthreshold,maxepochval,sampletheta)




# In[46]:


ojnparrbackup5 = np.copy(finojnparr)
thetanparrbackup5 = np.copy(finthetanparr)


# In[48]:


np.save('ojnparrbackup5',ojnparrbackup5)
np.save('thetanparrbackup5',thetanparrbackup5)


# In[45]:


# acc = tester(xarrtrain,yarrtrain,finalnetwork100,finojnparr100,finthetanparr100)
acc = tester(xarrtrain,yarrtrain,finalnetwork,finojnparr,finthetanparr)
print(acc)


# In[ ]:


tempforwardpass = forwardpass(xarrtrain[1],samplenetwork100)
# tempforwardpass = forwardpass(xarrtrain[1],samplenetwork)
print(tempforwardpass)
print(yarrtrain[0])
correct=0
wrong=0
for i in range(100):
# for i in range(numtrain):
    tempforwardpass= forwardpass(xarrtrain[i],samplenetwork100)
#     tempforwardpass= forwardpass(xarrtrain[i],samplenetwork)
    bool=1
    for j in range(len(yarrtrain[i])):
        tempval1=tempforwardpass[j]
        tempval2=0
        if(tempval1>0.5):
            tempval2=1
        else:
            tempval2=0
        if(tempval2==yarrtrain[i][j]):
            continue
        else:
            bool=0
            break
    if(bool==1):
        print("correct")
        correct+=1
    else:
        print("wrong")
        wrong+=1
print((1.0*correct)/(1.0*(correct+wrong)))
        


# In[49]:


templl = np.array([1,23,3]).reshape(3,1)
print(templl)
templl=np.append(templl,1).reshape(4,1)
print(templl)
templl= np.delete(templl,3,0)
print(templl)
templl=[1,2,3,4,5]
random.shuffle(templl)
print(templl)

