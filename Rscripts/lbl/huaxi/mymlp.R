x<-traindata[,c(Choose)]
y<-traindata[,ncol(traindata)]
xt<-testdata[,c(Choose)]
yt<-testdata[,ncol(testdata)]


aaa<-mlp(x,y)
rr<-predict(aaa,xt)


ttt<-roc(yt,as.numeric(rr))
plot(ttt)
