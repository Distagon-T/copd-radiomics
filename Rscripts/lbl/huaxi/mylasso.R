
mylasso<-function(x,y,xt,yt,m= "auc",n = 10, f="binomial",a=0.2){
  library(glmnet)
  lasso<-cv.glmnet(as.matrix(x),y,type.measure= m ,nfolds = n,family=f,alpha=a)
  lasso.best <- lasso$glmnet.fit #对应的最佳模型
  lasso.coef <- coef(lasso$glmnet.fit, s = lasso$lambda.1se) #

  coef<-lasso.coef[which(lasso.coef != 0)] #选择的变量系数
  plot(lasso)
  plot(lasso.best)
  choose<-names(lasso.coef[lasso.coef[,1]!=0,])#选择的变量名
  radio<-predict(lasso,as.matrix(x))
  #测试AUC
  radio2<-predict(lasso,as.matrix(xt))
  yt<-as.vector(yt)
  ROC.lasso<-roc(yt,as.numeric(radio2))
  plot(ROC.lasso)

  
  #绘图
  prob<-data.frame(yt)
  names(prob)[ncol(prob)]<-"Class"
  prob$prob<-radio2
  prob<-within(prob,{
    color<-NA
    color[Class==-1]<-"red"
    color[Class==1]<-"green"})
  prob<-prob[order(prob$prob),]
  
  t<-mean(prob$prob)
  prob$prob<-prob$prob-t    #取0.6作为阑值
  b<-t(prob$prob)
  b<-b[,1:ncol(b)]   #没这个报错为啥我也不知道
  barplot(b,col=prob$color,main="lasso回归")
  
  return(list(lasso=lasso,choose=choose,radio=radio,radio2=radio2,AUC= ROC.lasso))
  
}


