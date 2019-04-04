# one hot encoding
count=0
onehotxtest = []
for i in range(23):
    if(xidentifier[i]!=2):
        onehotxtest.append([])
        for j in range(numtest):
            onehotxtest[count].append(xarrtest[i][j])
        count+=1
    else:
        tempcat = xnumchildhelper[i]
        for j in range(tempcat):
            onehotxtest.append([])
        for j in range(numtest):
            tempval = xarrtest[i][j]
            for k in range(tempcat):
                if(k==tempval):
                    onehotxtest[count+k].append(1)
                else:
                    onehotxtest[count+k].append(0)
        count += tempcat
templen=len(onehotxtest)
onehotxtest = np.array(onehotxtest).reshape((templen,numtest))