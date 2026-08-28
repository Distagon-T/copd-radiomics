library(glmnet)
inputData <- read.csv(file="/Users/shenchen/Desktop/R/aLIEz/refined/Cross_Validation/3d_train_L.csv",header=T)

x <- data.frame(inputData[,1:9])
y <- as.factor(inputData[,10])

lasso <- cv.glmnet(as.matrix(x),y,type.measure= "auc" ,nfolds = 10,family="binomial",alpha=1)

lasso.best <- lasso$glmnet.fit #??Ӧ??????ģ??
lasso.coef <- coef(lasso$glmnet.fit, s = lasso$lambda.1se) #

coef<-lasso.coef[which(lasso.coef != 0)] #ѡ???ı?��ϵ??
plot(lasso)

radio<-predict(lasso,as.matrix(x))

library(pROC)
ROC<-roc(y,as.vector(radio))
plot(ROC)

testData <- read.csv(file="/Users/shenchen/Desktop/R/aLIEz/refined/Cross_Validation/3d_test_L.csv",header=T)

xt <- data.frame(testData[,1:9])
yt <- as.vector(testData[,10])

radio2<-predict(lasso,as.matrix(xt))
ROCt<-roc(yt,as.vector(radio2))
plot(ROCt)

