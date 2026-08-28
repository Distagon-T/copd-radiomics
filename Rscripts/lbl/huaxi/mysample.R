mysample<-function(data,f=2){
names(data)[ncol(data)]<-"Class"

l<-levels(factor(data$Class))
Adata<-subset(data,Class==l[1])
Bdata<-subset(data,Class==l[2])
a<-nrow(Adata)
b<-nrow(Bdata)
c<-(a-b)
c
if(f==1)     #过采样
  {
if(c<0){
  x<-sample(nrow(Adata),b,replace = T)
  sdata<-Adata[x,]
  newdata<-rbind(sdata,Bdata)
}else{
  x<-sample(nrow(Bdata),a,replace = T)
  sdata<-Bdata[x,]
  newdata<-rbind(sdata,Adata)}
}
else if(f==2)   #欠抽样
{
  if(c<0){
    x<-sample(nrow(Bdata),a,replace = T)
    sdata<-Bdata[x,]
    newdata<-rbind(sdata,Adata)
  }else{
    x<-sample(nrow(Adata),b,replace = T)
    sdata<-Bdata[x,]
    newdata<-rbind(sdata,Bdata)}
}
else{
  stop("wrong input")
}
return(newdata)
}