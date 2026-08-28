rm(list=ls(all=TRUE))

library(e1071)

##### train ######
inputData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/Cross_Validation/3d_train_L.csv",header=T)

x <- data.frame(inputData[,1:9])
y <- as.factor(inputData[,10])


t<- tune.svm(x,y, gamma = 10^(-4:1), cost = 2^(1:8),kernel="radial") # tune
b<-t$best.parameters
g<-b[1]
c<-b[2]
SVM<-svm(x,y,gamma=g,cost=c,kernel="radial",probability = TRUE,cross=10)
print(c)
print(g)

decision<-as.numeric(SVM$decision.values)  #??decision value???��?
radio2<-predict(SVM,x,decision.values=TRUE)


library(pROC)
ROC<-roc(y,decision)
plot(ROC)

##### test #######

testData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/guangdong_3d_refined_L.csv",header=T)

xt <- data.frame(testData[,1:9])
yt <- as.vector(testData[,10])

radio3 <- predict(SVM,xt,decision.values=TRUE)
decision_t <- as.numeric(attributes(radio3)$decision.value)

ROCt<-roc(yt,decision_t)
plot(ROCt)

radiomics <- sign(decision_t)