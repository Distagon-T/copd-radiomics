# Setup
library(e1071)
data(cats,package="MASS")
inputData <- data.frame(cats[,c(2,3)] ,response =as.factor(cats$Sex))

# linear SVM
svmfit <- svm(response ~ ., data=inputData, kernel="linear",cost=10,scale=FALSE)
plot(svmfit,inputData)

compareTable <- table (inputData$response, predict(svmfit))  # tabulate

print(compareTable)

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

svmfit <- svm (response ~ ., data = trainingData, kernel = "radial", cost = 100, gamma=0.001, scale = FALSE) # radial svm, scaling turned OFF
print(svmfit)
plot(svmfit, trainingData)
compareTable <- table (testData$response, predict(svmfit, testData))  # comparison table
mean(testData$response != predict(svmfit, testData)) # 13.79% misclassification error