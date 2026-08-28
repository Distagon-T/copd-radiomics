rm(list=ls())

# install.packages("car")
# install.packages("h2o")
# install.packages("statmod")
# install.packages("rms")
# install.packages("glmnet")
# install.packages("pROC")
# install.packages("ggplot2")
# install.packages("Hmisc")
# install.packages("stats")
# install.packages("randomForest")
# install.packages("caret")
# install.packages("ROCR")
# install.packages("e1071")
# install.packages("gam")
# install.packages("randomForest")
# install.packages("mRMRe")
# install.packages("DMwR")
# install.packages("rpart")
# install.packages("RSNNS")

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

setwd("C:/Users/zhao/OneDrive/文档/huaxi")
source("mysbf.R")
source("myheatmap.R")
source("mynomo.R")
source("mylasso.R")
source("mysvm.R")
source("mysample.R")
source("mymrmre.R")
source("myjudge.R")
source("myada.R")

data<-read.csv(file="E:/yunpan/work/program_radiomics/data/data.csv") 
data2<-read.csv(file="E:/yunpan/work/program_radiomics/data/info.csv") 
#data2<-read.csv(file.choose())
newdata<-merge(data,data2,by="Number")  #按ID整合数据


names(newdata)
newdata<-newdata[,2:498]
newdata$T<-substr(newdata$TNM,2,2)
newdata$N<-substr(newdata$TNM,4,4)
newdata$M<-substr(newdata$TNM,6,6)
newdata$M<-as.factor(newdata$M)
newdata$N<-as.factor(newdata$N)
newdata$T<-as.factor(newdata$T)
newdata$Age<-as.numeric(newdata$Age)
newdata$Smoke<-as.numeric(newdata$Smoke)
newdata$病理诊断<-as.factor(newdata$病理诊断)

newdata[newdata=="X"]<-NA
newdata[newdata=="O"]<-0
newdata<-within(newdata,{
  canner<-NA
  canner[病理诊断=="腺癌"]<-1
  canner[病理诊断=="鳞癌"]<-0})
newdata$ROS.1
newdata<-within(newdata,{
  ALKV<-NA
  ALKV[ALK.V=="阳性"]<-1
  ALKV[ALK.V=="可疑阳性"]<-1
  ALKV[ALK.V=="阴性"]<--1
  ALKV[ALK.V=="未做"]<-0})


newdata<-within(newdata,{
  ROS<-NA
  ROS[ROS.1=="可疑阳性"]<-1
  ROS[ROS.1=="阳性"]<-1
  ROS[ROS.1=="灶性弱阳"]<-1
  ROS[ROS.1=="灶性阳性"]<-1
  ROS[ROS.1=="阴性"]<--1
  ROS[ROS.1=="未做"]<-0})

newdata<-within(newdata,{
  Class<-NA
  Class[N==1|2|3]<-1
  Class[N==0]<--1})

names(newdata)
newdata$Time <- as.Date(newdata$时间, format="%Y-%m-%d")
newdata<-newdata[order(newdata$Time),]        #按时间排序
names(newdata)
newdata2<-newdata[,c(1:485,487:489,504)]#1-485 radio特征 487-489 报告特征 504 淋巴结侵犯
newdata2<-na.omit(newdata2)        #清理空值
summary(newdata2)
a<-names(newdata2)
name1<-a[1:161]  #可视特征
name2<-a[162:485] #不可视

newdata2[,1:485]<-scale(newdata2[,1:485], center = TRUE, scale = TRUE)  #radio standard



##############训练集测试集####################
traindata<-newdata2[1:199,]      #按时间分测试集与训练集
testdata<-newdata2[200:nrow(newdata2),]

#均衡
traindata<-mysample(traindata) #这里以最后一列作为分类

x<- cor(traindata[,1:485])  #影像特征
kappa(x, exact=T)   #共线性检查

#################################heatplot#########################################

# library(gplots)
# #dev.off()
# myheatmap(traindata[,c(1:485)],traindata[,ncol(traindata)]) 

############################lasso###################################

# #降维
 Choose<-mysbf(traindata[,1:485],traindata[,ncol(traindata)])
Choose
#降维2
# C<-mymrmre(traindata[,1:485],traindata[,ncol(traindata)],f=5,s=1)
# Choose<-C$Choose
ada
x<-traindata[,c(Choose)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose)]
yt<-testdata[,ncol(testdata)]


ada<-myada(x,y,xt,yt,n=50)


#lasso
ml<-mylasso(x,y,xt,yt)
radio<-ml$radio
radio2<-ml$radio2
#nomo
x1<-cbind(radio,traindata[487:488])   #结合报告特征
y1<-traindata[489]
names(x1)[1]<-"Radio" #命名
x2<-cbind(radio2,testdata[487:488])
y2<-testdata[489]
names(x2)[1]<-"Radio"

mn<-mynomo(x1,y1,x2,y2)
mn
############################visual+unvisual###############################

#lasso1
Choose1<-mysbf(traindata[,c(name1)],traindata[,ncol(traindata)])
x<-traindata[,c(Choose1)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose1)]
yt<-testdata[,ncol(testdata)]
ml1<-mylasso(x,y,xt,yt)
radio<-ml1$radio
tradio<-ml1$radio2

#lasso2
Choose2<-mysbf(traindata[,c(name2)],traindata[,ncol(traindata)])
x<-traindata[,c(Choose2)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose2)]
yt<-testdata[,ncol(testdata)]
ml2<-mylasso(x,y,xt,yt)
radio2<-ml2$radio
tradio2<-ml2$radio2

#nomo
x1<-cbind(radio,radio2,traindata[487:488])   #结合报告特征
y1<-traindata[ncol(traindata)]
names(x1)[1]<-"visual" #命名
names(x1)[2]<-"texture" #命名

x2<-cbind(tradio,tradio2,testdata[487:488])
y2<-testdata[ncol(traindata)]
names(x2)[1]<-"visual" #命名
names(x2)[2]<-"texture" #命名
mn2<-mynomo(x1,y1,x2,y2)


################################SVM#########################################
x<-traindata[,c(Choose)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose)]
yt<-testdata[,ncol(testdata)]
xt<-x
yt<-y
t<-mySVM(x,y,xt,yt)
radio<-as.numeric(t$radio)
radio2<-as.numeric(t$radio2)
#nomo
x1<-cbind(radio,traindata[487:488])   #结合报告特征
y1<-traindata[489]
names(x1)[1]<-"Radio" #命名

x2<-cbind(radio2,testdata[487:488])
y2<-testdata[489]
names(x2)[1]<-"Radio"
x1
mn<-mynomo(x1,y1,x2,y2)

##########