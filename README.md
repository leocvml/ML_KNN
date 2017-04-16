# ML_KNN   
 my first ML_Algorithm
 KNN
  
1.calculate the distance between input and the current data  
2.sort the distances in increasing order  
3.take k item with lowest distances to input  
4.finnd majority class among these item  
5.return majority class as out predict for the class of input  
  

# how to use kNN.py  
## 1.import kNN module  
**def createDataSet():**  
``will return group,labels``  
**def classify0(idx,dataSet,labels,k):**  
``predict your input with dataSet using Ｅuclid distance  ,  k  = vote``    
**def file2matrix(filename):**  
``file to matrix``    
**def show_data2img(DataMat,DataLabels,datatype1,datatype2):**  
    ``data visualization``    
**def autoNorm(dataSet):**  
    ``normalization``    
**def datingClassTest():**  
    ``Test your model and observation your errorRate``    
**def classifyPerson():**  
    ``verify``    
**def img2vector(filename):**  
    ``img 2 32*32 vector``   
**def handwritingClassTest():**  
    ``Test handwriting model``     

          

## 2.read data set   
## 3. Test your model  
  **classify0(dataSet) observation errorrate**  
## 4. use your model in real  




# Sample  
1. dating  
``import kNN``     
``datingMat , datingLabels = file2matrix('datingTestSet.txt')``  
``kNN.show_data2img(datingMat,datingLabels , 0 ,1)    ==> Data visualization``  
``kNN.datingClassTest()``  
``kNN.classifyPerson()``    


2. handwriting recognition  
``import kNN``  
``kNN.handwritingClassTest()``  









