import numpy as np
import math
import random
import sys
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
def neuralnet(inparr, outarr, hiddeninfo,learningrate, batchsize,costthres,maxepochallowed,sampletheta):
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
    #            if(batchsize!=1):
     #              random.shuffle(shufflehelper)

                costarr, newavgcost=findtotalcostarr(inparr,outarr,network, ojnparr,thetanparr)
                
                if(debug0==1): print("cost is")
                if(debug0==1): print(newavgcost)
                if(oldavgcost-newavgcost<costthres):
                    print("oldcost is "+str(oldavgcost))
                    print("newcost is "+str(newavgcost))
                    print("cost is lower")
                    break
                elif(numwholedatapass>=maxepochallowed):
                    break
                else:
                    #do nothing continue
                    oldavgcost=newavgcost
                
        
            
            
            
#         tempforwardpass = forwardpass(inparr[0],network)
#         print(tempforwardpass)
#         #lets see cost
#         print("cost = "+str(findcost(tempforwardpass,outarr[0])))        
    
    return network,thetanparr
sampleinparr = [[0.05, 0.10]]
# sampleinparr = [[0.0, 1.0],[1.0,0.0],[0.0,0.0],[1.0,1.0]]
# sampleinparr = [[0.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]]
sampleoutput = [[0.01, 0.99]]
# sampleoutput = [[0.99],[0.99],[0.01],[0.99]]
# sampleoutput=[[1.0,0.0],[1.0,0.0],[1.0,0.0],[1.0,0.0],[0.0,1.0],[0.0,1.0],[0.0,1.0],[0.0,1.0]]
samplehiddeninfo = [2]
samplelearningrate=0.1
sampletheta = [[[.15,.20,.35],[.25,.30,.35]],[[.40,.45,.60],[.50,.55,.60]]]

samplenetworksample,thetanparr = neuralnet(sampleinparr,sampleoutput,samplehiddeninfo,samplelearningrate,1, pow(10,-5),1000,sampletheta)
print(samplenetworksample)
            

    
