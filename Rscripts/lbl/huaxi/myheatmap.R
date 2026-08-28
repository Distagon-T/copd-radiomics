myheatmap<-function(x,y){
  traindata<-cbind(x,y)
  names(traindata)[ncol(traindata)]<-"Class"
  traindata2<-within(traindata,{
    color<-"#8CFF00FF"
    color[Class==1]<-"#8CFF00FF"
    color[Class==0]<-"#FF1E00FF" })
  traindata2<-traindata2[,c(1:485,ncol(traindata2))]
  heatmap.2(as.matrix(traindata2[1:485]), scale="column", 
            RowSideColors=traindata2[,ncol(traindata2)],margins=c(5,10),trace="none")
  
}