import numpy as np
import sys
trainfname = open(sys.argv[1], 'r')
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

testfname = open(sys.argv[2], 'r')
xarrtestori=[]
yarrtestori=[]
numtest=0
for line in testfname:
    linearr=line.split(',')
    xarrtestori.append([])
    yarrtestori.append([])
    for i in range(len(linearr)-1):
        xarrtestori[numtest].append(int(linearr[i]))
    yarrtestori[numtest].append(int(linearr[len(linearr)-1]))
    numtest+=1
xnumchildhelper = [4,13,4,13,4,13,4,13,4,13]
ynumchildhelper = [10]
print(len(xarrtestori[0]))


#one hot encoding
xnumfeatures=0
xarrtest = []
for i in range(len(xarrtestori[0])): 
    tempcat = xnumchildhelper[i]
    for j in range(tempcat):
        xarrtest.append([])
    for j in range(numtest):
        tempval = xarrtestori[j][i]
        for k in range(tempcat):
            if(k==tempval):
                xarrtest[xnumfeatures+k].append(1.0)
            else:
                xarrtest[xnumfeatures+k].append(0.0)
    xnumfeatures += tempcat
print("done")
# print(len(xarrtest))
# print(xnumfeatures)
xarrtest = np.array(xarrtest).reshape((xnumfeatures,numtest))
xarrtest = np.transpose(xarrtest)
xarrtest = xarrtest.tolist()
# print(len(xarrtest))
# print(len(xarrtest[0]))

ynumfeatures=0
yarrtest = []
for i in range(len(yarrtestori[0])): 
    tempcat = ynumchildhelper[i]
    for j in range(tempcat):
        yarrtest.append([])
    for j in range(numtest):
        tempval = yarrtestori[j][i]
        for k in range(tempcat):
            if(k==tempval):
                yarrtest[ynumfeatures+k].append(1.0)
            else:
                yarrtest[ynumfeatures+k].append(0.0)
    ynumfeatures += tempcat
print("done")
# print(len(yarrtest))
# print(ynumfeatures)
yarrtest = np.array(yarrtest).reshape((ynumfeatures,numtest))
yarrtest = np.transpose(yarrtest)
yarrtest = yarrtest.tolist()
# print(len(yarrtest))
# print(len(yarrtest[0]))


trainwfile = open(sys.argv[3],"w")
trainwfile.write(str(numtrain)+"\n")
trainwfile.write(str(len(xarrtrain[0]))+"\n")
dimx = len(xarrtrain[0])
for i in range(numtrain):
	temp=""
	for j in range(dimx):
		if(j==dimx-1):
			temp += str(xarrtrain[i][j])+"\n"
		else:
			temp += str(xarrtrain[i][j])+","
	trainwfile.write(temp)
trainwfile.write(str(len(yarrtrain[0]))+"\n")
dimy = len(yarrtrain[0])
for i in range(numtrain):
	temp=""
	for j in range(dimy):
		if(j==dimy-1):
			temp += str(yarrtrain[i][j])+"\n"
		else:
			temp += str(yarrtrain[i][j])+","
	trainwfile.write(temp)
trainwfile.close()

testwfile = open(sys.argv[4],"w")
testwfile.write(str(numtest)+"\n")
testwfile.write(str(len(xarrtest[0]))+"\n")
dimx = len(xarrtest[0])
for i in range(numtest):
	temp=""
	for j in range(dimx):
		if(j==dimx-1):
			temp += str(xarrtest[i][j])+"\n"
		else:
			temp += str(xarrtest[i][j])+","
	testwfile.write(temp)
testwfile.write(str(len(yarrtest[0]))+"\n")
dimy = len(yarrtest[0])
for i in range(numtest):
	temp=""
	for j in range(dimy):
		if(j==dimy-1):
			temp += str(yarrtest[i][j])+"\n"
		else:
			temp += str(yarrtest[i][j])+","
	testwfile.write(temp)
testwfile.close()
print("written")
