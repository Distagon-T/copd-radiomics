myada<-function(x,y,xt,yt,n=100){
  w<-rep(1/nrow(x),time=nrow(x))
  traindata<-cbind(x,y)
  names(traindata)[ncol(traindata)]<-"Class"
  weaklearn<-list()
  al<-list()
  for(i in 1:n){
    print(i)  #看进度
    weakL<-rpart(Class~.,method="class",weight=w,data=traindata) 
    a<-(as.numeric(predict(weakL,x,type = "class"))-1.5)*2
    c<-myjudge(a,y)
    e<-sum(c*w)
    alpha<-0.5*log((1-e)/e)
    z<-sum(w*exp(-alpha*y*a))
    w<-w*exp(-alpha*y*a)/z

    weaklearn<-c(weaklearn,list(weakL))   #存储弱分类器
    al<-c(al,list(alpha))                 #存储alpha值
    
    zzz<-matrix(nrow=nrow(x),ncol=i)  
    zz<-0
    testdata<-cbind(xt,yt)
    names(testdata)[ncol(testdata)]<-"Class"
    for(j in 1:i){
      test<-weaklearn[[j]]
      alp<-al[[j]]
      tt<-(as.numeric(predict(test,x,type = "class"))-1.5)*2
      zzz[,j]<-as.numeric(alp)*tt
      zz<-zz+zzz[,j]
    }
    t<-sum(abs(y-sign(zz)))/2
    if(t==0){
      break
      # return(weaklearn)
      # return(al)
    }
  }
  
  l2<-length(weaklearn)
  aa2<-0
  aaa2<-matrix(nrow=nrow(traindata),ncol=l2)
  for(j in 1:l2){
    test2<-weaklearn[[j]]
    tw2<-al[[j]]
    tt2<-(as.numeric(predict(test2,x,type = "class"))-1.5)*2
    aaa2[,j]<-as.numeric(tw2)*tt2
    aa2<-aa2+aaa2[,j]
  }
  
  l<-length(weaklearn)
  aa<-0
  aaa<-matrix(nrow=nrow(testdata),ncol=l)
  for(j in 1:l){
    test<-weaklearn[[j]]
    tw<-al[[j]]
    tt<-(as.numeric(predict(test,xt,type = "class"))-1.5)*2
    aaa[,j]<-as.numeric(tw)*tt
    aa<-aa+aaa[,j]
    print(aa)
  }
  sss<-testdata[,"Class"]
  asd<-sum(abs(sss-sign(aa)))/2
  return(list(acc=(1-(asd/nrow(testdata))),weight=aa2,weight2=aa,abs=sign(aa)))
}


