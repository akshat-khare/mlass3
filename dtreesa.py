#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
trainfname = open(sys.argv[1], 'r')
xarrtrainori = []
yarrtrain = []
xidentifier= [3,1,2,2,3,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3]
medianarr = [0.0]*23
xnumchildhelper=[]
for i in range(23):
    if(xidentifier[i]!=2):
        xnumchildhelper.append(2)
    else:
        if(i==2):
            xnumchildhelper.append(7)
        elif(i==3):
            xnumchildhelper.append(4)
        elif(i>=5 and i<=10):
            xnumchildhelper.append(12)
print(xnumchildhelper)
# 1 means binary, 2 means categorical, 3 means continous
for i in range(23):
    xarrtrainori.append([]);
numtrain=0
for line in trainfname:
    numtrain+=1
    if(numtrain<=2):
        continue
    linearr= line.split(',')
#     print(linearr)
    for i in range(23):
        xarrtrainori[i].append(int(linearr[i+1]))
    yarrtrain.append(int(linearr[24]))
#     if(numtrain>600):
#         break
numtrain = numtrain-2
print("parsing done")
# print(xarrtrainori)
# print("y is")
# print(yarrtrain)


# In[2]:


valifname = open(sys.argv[2], 'r')
xarrvaliori = []
yarrvali = []
for i in range(23):
    xarrvaliori.append([]);
numvali=0
for line in valifname:
    numvali+=1
    if(numvali<=2):
        continue
    linearr= line.split(',')
#     print(linearr)
    for i in range(23):
        xarrvaliori[i].append(int(linearr[i+1]))
    yarrvali.append(int(linearr[24]))
#     if(numtrain>600):
#         break
numvali = numvali-2
print("parsing done")
# print(xarrtrainori)
# print("y is")
# print(yarrtrain)


# In[3]:


testfname = open(sys.argv[3], 'r')
xarrtestori = []
yarrtest = []
for i in range(23):
    xarrtestori.append([]);
numtest=0
for line in testfname:
    numtest+=1
    if(numtest<=2):
        continue
    linearr= line.split(',')
#     print(linearr)
    for i in range(23):
        xarrtestori[i].append(int(linearr[i+1]))
    yarrtest.append(int(linearr[24]))
#     if(numtrain>600):
#         break
numtest = numtest-2
print("parsing done")
# print(xarrtrainori)
# print("y is")
# print(yarrtrain)


# In[4]:


xarrtrain=[]
for i in range(23):
    xarrtrain.append([])
for i in range(23):
    if(i>=5 and i<=10):
        for j in range(numtrain):
            xarrtrain[i].append(xarrtrainori[i][j]+2)
    elif(i==1):
        for j in range(numtrain):
            xarrtrain[i].append(xarrtrainori[i][j]-1)
    elif(xidentifier[i]!=3):
        for j in range(numtrain):
            xarrtrain[i].append(xarrtrainori[i][j])
    else:
        templist=[]
        for j in range(numtrain):
            templist.append(xarrtrainori[i][j])
#         templist = xarrtrainori[i]
        templist.sort()
#         print(templist)
        median=0.0
        if(numtrain%2==1):
            median=templist[int(numtrain/2)]
        else:
            median = (0.5*(templist[int(numtrain/2)] + templist[int(numtrain/2)-1]))
        medianarr[i] = median
        print("median for "+str(i) + " is "+ str(median))
        for j in range(numtrain):
            if(xarrtrainori[i][j]>median):
                xarrtrain[i].append(1)
            else:
                xarrtrain[i].append(0)
# print(xarrtrain)
# print(yarrtrain)


# In[5]:


xarrvali=[]
for i in range(23):
    xarrvali.append([])
for i in range(23):
    if(i>=5 and i<=10):
        for j in range(numvali):
            xarrvali[i].append(xarrvaliori[i][j]+2)
    elif(i==1):
        for j in range(numvali):
            xarrvali[i].append(xarrvaliori[i][j]-1)
    elif(xidentifier[i]!=3):
        for j in range(numvali):
            xarrvali[i].append(xarrvaliori[i][j])
    else:
#         templist=[]
#         for j in range(numvali):
#             templist.append(xarrvaliori[i][j])
# #         templist = xarrvaliori[i]
#         templist.sort()
#         print(templist)
        median=medianarr[i]
#         if(numvali%2==1):
#             median=templist[int(numvali/2)]
#         else:
#             median = (0.5*(templist[int(numvali/2)] + templist[int(numvali/2)+1]))
        print("median for "+str(i) + " is "+ str(median))
        for j in range(numvali):
            if(xarrvaliori[i][j]>median):
                xarrvali[i].append(1)
            else:
                xarrvali[i].append(0)
# print(xarrvali)
# print(yarrvali)


# In[6]:


import numpy as np
xarrvalinp = np.array(xarrvali).reshape((23,numvali))
xarrvalinp = np.transpose(xarrvalinp)


# In[7]:


xarrtest=[]
for i in range(23):
    xarrtest.append([])
for i in range(23):
    if(i>=5 and i<=10):
        for j in range(numtest):
            xarrtest[i].append(xarrtestori[i][j]+2)
    elif(i==1):
        for j in range(numtest):
            xarrtest[i].append(xarrtestori[i][j]-1)
    elif(xidentifier[i]!=3):
        for j in range(numtest):
            xarrtest[i].append(xarrtestori[i][j])
    else:
#         templist=[]
#         for j in range(numtest):
#             templist.append(xarrtestori[i][j])
# #         templist = xarrtestori[i]
#         templist.sort()
#         print(templist)
        median=medianarr[i]
#         if(numtest%2==1):
#             median=templist[int(numtest/2)]
#         else:
#             median = (0.5*(templist[int(numtest/2)] + templist[int(numtest/2)+1]))
        print("median for "+str(i) + " is "+ str(median))
        for j in range(numtest):
            if(xarrtestori[i][j]>median):
                xarrtest[i].append(1)
            else:
                xarrtest[i].append(0)
# print(xarrtest)
# print(yarrtest)


# In[8]:


# print(xarrtrain[4])
# print(xarrtrainori[0])


# In[9]:


debug=0
# 1 is true 0 is false
import math
def printtree(thisnode):
    print(thisnode.xsplit)
    print("|")
    if(thisnode.childlist==None):
        print("None is child")
        return
    if(len(thisnode.childlist)==0):
        print("Leaf "+str(thisnode.yleaf))
    for i in thisnode.childlist:
        printtree(i)
        print("--")

class Node:
    def __init__(self,ylist,target):
        self.ylist=[]
        for i in range(len(ylist)):
            (self.ylist).append(ylist[i])
        self.childlist=None
        self.target=[]
        for i in range(len(target)):
            (self.target).append(target[i])
        self.xsplit=[]
        self.yleaf=None
        self.spliton=None
        self.splitvalue=None
        for i in range(23):
            (self.xsplit).append(-1)
    def setchild(self,childlist):
        self.childlist=[]
        for i in range(len(childlist)):
            (self.childlist).append(childlist[i])
#             print("appending")
#             printtree(self)
#         self.childlist=childlist
    def updatexsplit(self,index,val):
        (self.xsplit)[index]=val
    def setxsplit(self,xsplit):
        for i in range(len(xsplit)):
            (self.xsplit)[i]=xsplit[i]
    def setyleaf(self,val):
        self.yleaf=val
    def setspliton(self,val):
        self.spliton=val
    def setsplitval(self,val):
        self.splitvalue= val
def entropy(arr):
    temp=0.0
    sumarr=(arr[0]+arr[1])*1.0
    if((arr[0]+arr[1])==0):
        if(debug==1): print("Zero Zero case in entropy----------")
        return math.log(2)/math.log(math.exp(1))
    elif(arr[0]==0 or arr[1]==0):
        if(debug==1): print("Zero one or one zero entropy case------")
        return 0.0
    for i in range(2):
        temp += -((1.0*arr[i])/sumarr)*((math.log((1.0*arr[i])/sumarr))/math.log(math.exp(1)))
    return temp
  
def test(arr,thisnode):
    if(len(thisnode.childlist)==1):
        return thisnode.childlist[0].yleaf
    else:
        temp = thisnode.spliton
        return test(arr, thisnode.childlist[arr[temp]])


def choosebestattr(thisnode):
    itarr =[]
    for i in range(23):
        if((thisnode.xsplit[i])>=0):
            itarr.append(float('-inf'))
            continue
        tempinf=[]
        numtarget=len(thisnode.target)
        yattarr=[]
        numattr=[]
        for j in range(xnumchildhelper[i]):
            yattarr.append([0,0])
            numattr.append(0)
        for j in thisnode.target:
            tempk = xarrtrain[i][j]
            numattr[tempk]=numattr[tempk]+1
            if(yarrtrain[j]==0):
                yattarr[tempk][0]=yattarr[tempk][0]+1
            else:
                yattarr[tempk][1]=yattarr[tempk][1]+1
        for j in range(xnumchildhelper[i]):
            temp = (   (1.0*(numattr[j])) / (1.0*numtarget) )   *   (entropy(yattarr[j])    ) 
            tempinf.append(temp)
        temp=0.0
        for j in range(xnumchildhelper[i]):
            temp+=tempinf[j]
        if(debug==1): print("hb is")
        if(debug==1): print(temp)
        temp = entropy(thisnode.ylist)-temp
#         print(entropy(thisnode.ylist))
        itarr.append(temp)
    tempval=itarr[0]
    maxone=0
    for i in range(23):
        if(itarr[22-i]>tempval):
            tempval=itarr[22-i]
            maxone=22-i
    if(debug==1): print(itarr)
    if(debug==1): print("max inf gain is "+str(tempval))
    return maxone
import random  
def allfeatureexplored(thisnode):
    temp=0
    for i in range(23):
        if(thisnode.xsplit[i]<0):
            #False
            temp+=1
#             return 1
    if(temp<=0):
        # loltemp = random.random()
        # # print(loltemp)
        # if(loltemp>0.8):
        #     return 0
        # else:
        #     return 1
        return 0
    else:
        return 1
    #True
#     return 0
                    
def grownode(thisnode):
    if(thisnode.ylist[0]==0):
        tempnode = Node(thisnode.ylist,thisnode.target)
        tempnode.setchild([])
        tempnode.yleaf=1
        tempnode.setxsplit(thisnode.xsplit)
        temp=[]
        temp.append(tempnode)
        thisnode.setchild(temp)
        return
    elif(thisnode.ylist[1]==0):
        tempnode = Node(thisnode.ylist,thisnode.target)
        tempnode.setchild([])
        tempnode.yleaf=0
        tempnode.setxsplit(thisnode.xsplit)
        temp=[]
        temp.append(tempnode)
        thisnode.setchild(temp)
        return
    elif(allfeatureexplored(thisnode)==0):
        tempnode = Node(thisnode.ylist,thisnode.target)
        tempnode.setchild([])
        if(tempnode.ylist[1]>tempnode.ylist[0]):
            tempnode.yleaf=1
        else:
            tempnode.yleaf=0
        tempnode.setxsplit(thisnode.xsplit)
        temp=[]
        temp.append(tempnode)
        thisnode.setchild(temp)
        return
    else:
        bestattr=choosebestattr(thisnode)
        if(debug==1): print("best attr is "+str(bestattr))
        tempnumchild = xnumchildhelper[bestattr]
        tempchildarr=[]
        for i in range(tempnumchild):
            temptarget=[]
            tempylist=[0,0]
            for j in thisnode.target:
                if(xarrtrain[bestattr][j]==i):
                    temptarget.append(j)
                    if(yarrtrain[j]==0):
                        tempylist[0]= tempylist[0]+1
                    else:
                        tempylist[1]= tempylist[1]+1
            tempnode = Node(tempylist,temptarget)
            tempnode.setxsplit(thisnode.xsplit)
            tempnode.updatexsplit(bestattr,i)
#             print("bestarr is "+str(bestattr))
#             print("i is "+str(i))
#             print(tempnode.xsplit)
#             print(i)
#             print(tempylist)
            grownode(tempnode)
            tempchildarr.append(tempnode)
#         print("before setting child")
#         printtree(thisnode)
        thisnode.setspliton(bestattr)
        thisnode.setchild(tempchildarr)
#         print("after setting child")
#         printtree(thisnode)
        return
                
tempysplit=[0,0]
temptarget=[]
for i in range(numtrain):
    temptarget.append(i)
    if(yarrtrain[i]==0):
        tempysplit[0]= tempysplit[0]+1
    else:
        tempysplit[1]= tempysplit[1]+1
root = Node(tempysplit,temptarget)
grownode(root)
print("done")
if(debug==1): printtree(root)


# In[10]:


debug=0
# print(len(root.childlist))
# test([1]*23,root)
numright=0
numwrong=0
for i in range(numtrain):
    temp=[]
    for j in range(23):
        temp.append(xarrtrain[j][i])
    ypred = test(temp,root)
    if(debug==1): print(ypred)
    if(ypred==yarrtrain[i]):
        if(debug==1): print("right")
        numright+=1
    else:
        if(debug==1): print("wrong")
        numwrong+=1
# print("hell")


# In[11]:


print((numright*1.0)/(1.0*numtrain))


# In[12]:


# test([1]*23,root)
numright=0
numwrong=0
for i in range(numvali):
    temp=[]
    for j in range(23):
        temp.append(xarrvali[j][i])
    ypred = test(temp,root)
#     print(ypred)
    if(ypred!=0 and ypred!=1):
        print("error")
    if(ypred==yarrvali[i]):
#         print("right")
        numright+=1
    else:
#         print("wrong")
        numwrong+=1
# print("hell")


# In[13]:


print(numright)
print(numwrong)
print(numvali)
print((numright*1.0)/(1.0*numvali))


# In[14]:


# test([1]*23,root)
debug=0
numright=0
numwrong=0
for i in range(numtest):
    temp=[]
    for j in range(23):
        temp.append(xarrtest[j][i])
    ypred = test(temp,root)
    if(debug==1): print(ypred)
    if(ypred==yarrtest[i]):
        if(debug==1): print("right")
        numright+=1
    else:
        if(debug==1): print("wrong")
        numwrong+=1
# print("hell")


# In[15]:


print(numright)
print(numwrong)
print(numvali)
print((numright*1.0)/(1.0*numtest))

def numnodes(root):
    if((len(root.childlist))==1):
        return 1
    else:
        temp=0
        for i in root.childlist:
            temp += numnodes(i)
        return temp +1
print(numnodes(root))