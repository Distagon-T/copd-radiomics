library(RSNNS)
library(DMwR)
library(mRMRe)
library(randomForest)
library(caret)
library(stats)
library(h2o)
library(rms)
library(glmnet)
library(class)
library(pROC)
library(car)
library(ggplot2)
library(ROCR)
library(e1071)
library(gam)
library(gplots)
library(ggplot2)
library(rpart)

setwd("C:/Users/zhao/OneDrive/ÎÄµµ/huaxi")
source("mysbf.R")
source("myheatmap.R")
source("mynomo.R")
source("mylasso.R")
source("mysvm.R")
source("mysample.R")
source("mymrmre.R")
source("myjudge.R")
source("myada.R")
data1<-read.csv(file.choose(),header = F)
data2<-read.csv(file.choose(),header = F)
traindata<-data1
testdata<-data2

# #½µÎ¬
Choose<-mysbf(traindata[,-ncol(traindata)],traindata[,ncol(traindata)])

Choose

x<-traindata[,c(Choose)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose)]
yt<-testdata[,ncol(testdata)]
ada<-myada(x,y,xt,yt,n=50)
ada$weight
aaaa<-roc(y,ada$weight)
plot(aaaa)



mylasso(x,y,xt,yt)
ada<-myada(x,y,xt,yt,n=20)
ada$weight
aaaa<-roc(yt,ada$weight)
plot(aaaa)


sss<-mysvm(x,y,xt,yt)
aaaa<-roc(yt,ada$weight)
plot(aaaa)

ttt<-mlp(x,y)
ttt
zzz<-predict(ttt,xt)
plot(roc(yt,zzz))

