mysvm<-function(x,y,xt,yt,k="polynomial"){
t<-tune.svm(x,y,gamma=2^(-1:2),cost=2^(-1:4),probability=T,kernel=k)
b<-t$best.parameters
g<-b[1]
c<-b[2]
SVM<-svm(x,y,gamma=g,cost=c,kernel=k,probability = TRUE,cross=5)
radio<-SVM$decision.values  #ÓÃdecision value×öÆÀ·Ö
radio2<-predict(SVM,xt,decision.values=TRUE)
ROC.lasso<-roc(yt,radio2)
plot(ROC.lasso)
return(list(SVM<-SVM,radio=radio,radio2=radio2))
}


