
rm(list=ls())

traindata<-read.csv(file="/home/luviagelita/Documents/lbl/lbl_pre.csv",header=F)
testdata<-read.csv(file="/home/luviagelita/Documents/lbl/lbl_after.csv",header=F)

tdata<-scale(traindata[,-ncol(traindata)])
traindata<-cbind(tdata,traindata[,ncol(traindata)])
traindata<-data.frame(traindata)


Adata<-subset(traindata,V643==1,select=c(1:642))
Bdata<-subset(traindata,V643==0,select=c(1:642))
dim(Adata)
dim(Bdata)


#???????????ݿ?
Fname<-names(Adata)
length(Fname)
Features<-data.frame(Fname)
dim(Features)
data1
#heatmap0
data1<-traindata[,c(1:643)]
data1<-within(data1,{
  color<-NA
  color[V643==1]<-"#8CFF00FF"
  color[V643==0]<-"#FF1E00FF"})
library(gplots)
heatmap.2(as.matrix(data1[,c(1:642)]),RowSideColors=data1[,644],trace="none")


#t-test
p_value<-0
for (i in 1:length(Fname)){
  t<-t.test(Adata[,i],Bdata[,i])
  p_value[i]<-t$p.value
}
library(pROC)
Features$ttest_p<-p_value
AUC<-0

for (i in 1:length(Fname)){
  F<-roc(V643~traindata[,i],traindata)
  AUC[i]<-F$auc
}
Features$AUC<-AUC
Features<-Features[order(Features$AUC),]
FailureFeatures<-subset(Features,ttest_p>0.01)
vars<-names(traindata)%in%FailureFeatures[,1]
traindata<-traindata[!vars]
Features



set.seed(2601)   #??????ѭ???Գ?��??
x<-as.matrix(traindata[,c(318,593,7,414,75,197)])
y<-as.factor(traindata[,ncol(traindata)])
xt<-as.matrix(testdata[,c(318,593,7,414,75,197)])
yt<-as.factor(testdata[,ncol(testdata)])


lasso<-cv.glmnet(x,y,type.measure="mse",family="binomial",n=5)
names(lasso)
plot(lasso)

lasso.best <- lasso$glmnet.fit #??Ӧ??????ģ??
lasso.coef <- coef(lasso$glmnet.fit, s = lasso$lambda.1se) #ϵ??
lasso.coef[which(lasso.coef != 0)] #ѡ???ı?��
plot(lasso.best)

pred<-predict(lasso,xt)
yt<-as.numeric(yt)
pred
ROC<-roc(yt,as.numeric(pred),ci=TRUE)
plot(ROC)








