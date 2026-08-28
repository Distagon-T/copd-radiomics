################################nomogram######################################

mynomo<-function(x,y,xt,yt,fun=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),
                 funlabel="Risk",cmethod = 'KM', method = "boot", m = 100, B = 1000){
traindata<-cbind(x,y)
testdata<-cbind(xt,yt)
names(traindata)[ncol(traindata)]<-"Class"
#names(testdata)[ncol(testdata)]<-"Class"


###  这里是因为rms包datadist只能读取global所以加的几行让他在function里可以读的出来
on.exit(detach("design.options")) 
attach(list(), name="design.options") 
d <- data.frame(traindata) 
assign('dd', datadist(d), pos='design.options') 
options(datadist="dd") 




f <- lrm(Class~., data=traindata,x=TRUE, y=TRUE)    #建logistic模型

nom <- nomogram(f,fun=plogis,
                fun.at=fun,
                funlabel="Risk")
plot(nom)

validate(f, method="boot", B=1000, dxy=T)  #C-index值=（Dxy+1）/2
result<-rcorrcens(traindata[,ncol(traindata)] ~ predict(f), data = traindata) 
c1<-result[,1]+1.96*result[,4]/sqrt(result[,7]) #置信区间上界
c2<-result[,1]-1.96*result[,4]/sqrt(result[,7]) #置信区间下界


#验证曲线
cal<-calibrate(f, cmethod = 'KM', method = "boot", m = 100, B = 1000)  #验证曲线
plot(cal,scat1d.opts=list(nhistSpike=240,side=1,frac=0.08,lwd=1,nint=50))
lines(cal, lwd=2,lty=3,col=c(rgb(255,0,0,maxColorValue=255))) #给校正曲线上色
abline(0,1,lty =5,lwd=2,col=c(rgb(0,0,255,maxColorValue= 255))) 


# validation验证

fev<-lrm(testdata[,ncol(testdata)]~predict(f, newdata=testdata), x=T, y=T, data=testdata) 
validate(fev, method="boot", B=1000, dxy=T)
cal<-calibrate(fev, cmethod = 'KM', method = "boot", B = 1000) 
plot(cal,lwd=1,lty=1,errbar.col=c(rgb(0,0,0,maxColorValue = 255)),xlab ="Nomogram Predicted N",ylab="Actual N",col=c(rgb(255,0,0,maxColorValue =255)))
rcorrcens(testdata[,ncol(testdata)] ~ predict(fev), data = testdata)
result2<-rcorrcens(testdata[,ncol(testdata)] ~ predict(fev), data = testdata)
c3<-result2[,1]+1.96*result[,4]/sqrt(result[,7]) #置信区间上界
c4<-result2[,1]-1.96*result[,4]/sqrt(result[,7]) #置信区间下界
return(list(r1=result,r2=result2,c1=c1,c2=c2,c3=c3,c4=c4))
}
