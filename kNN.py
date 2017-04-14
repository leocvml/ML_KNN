from numpy import *
import operator

def createDataSet():
    group = array(([1.0,1.1],[1.0,1.0],[0,0],[0,0.1]))
    labels = ['A','A','B','B']
    return group,labels
def classify0(idx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(idx,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances =sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortDistIndicies = distances.argsort()
    classCount={}

    for i in range(k):
        voteIlabel = labels[sortDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)                              
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector =[]
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
def show_data2img(DataMat,DataLabels,datatype1,datatype2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DataMat[:,datatype1],DataMat[:,datatype2],15.0 * array(DataLabels) , 15.0 * array(DataLabels))
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals , (m,1))
    normDataSet = normDataSet / tile(ranges ,(m,1))
    return normDataSet , ranges , minVals

def datingClassTest():
    hotRatio = 0.99
    datingDataMat , datingLabels = file2matrix('datingTestSet.txt')
    normMat , ranges ,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    
    numTestVecs = int(m * hotRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("No.%d the classifier came back with: %d , the real answer is : %d" %(i+1,classfierResult , datingLabels[i]))
        if(classfierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / float(numTestVecs)))
    
    
