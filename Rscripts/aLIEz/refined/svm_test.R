test_data <- read.csv(file="/media/luviagelita/ubuntu/aLIEz/refined/NC_2d_refined_L.csv",header=T)

inputData <- data.frame(test_data[,1:9], response=as.factor(test_data[,10]))

library(e1071)
## Tuning
# Prepare training and test data
set.seed(100) # for reproducing results
rowIndices <- 1 : nrow(inputData) # prepare row indices
sampleSize <- 0.8 * length(rowIndices) # training sample size
trainingRows <- sample (rowIndices, sampleSize) # random sampling
trainingData <- inputData[trainingRows, ] # training data
testData <- inputData[-trainingRows, ] # test data
tuned <- tune.svm(response ~., data = trainingData, gamma = 10^(-6:-1), cost = 10^(1:2)) # tune
summary (tuned) # to select best gamma and cost

svmfit <- svm(response ~ .,data=inputData, kernel="radial",cost=100,gamma=0.01,scale=FALSE)
print(svmfit)
plot(svmfit, trainingData)
compareTable <- table (inputData$response, predict(svmfit, inputData))  # comparison table
mean(inputData$response != predict(svmfit, inputData)) # 13.79% mcisclassification error

library(pROC)

pred<-predict(svmfit,inputData[,1:9])
pred<-as.numeric(pred)
y<-as.numeric(inputData$response)
ROC.svm<-roc(y,pred)
plot(ROC.svm)