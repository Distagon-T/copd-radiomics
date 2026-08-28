mysbf<-function(x,y){
  # x is feature
  # y is response variable
  
  
  j<-1
  for(i in 1:ncol(x)){  #删除同列相同值
    t<-x[1,j]
    if(all(x[,j]==t)){
      x<-x[,-j]
      j<-j-1
    }
    j<-j+1
  }
  
  dd<-rbind(x,y)  
  ttt<-x
  ytt<-y
  descrCorr <- cor(ttt)
  highCorr <- findCorrelation(descrCorr, 0.90)
  ttt2 <- ttt[, -highCorr]
  # 数据预处理步骤（标准化，缺失值处理）
  Process <- preProcess(ttt2)
  ttt3 <- predict(Process, ttt2)
  # 用sbf函数实施过滤方法，这里是用随机森林来评价变量的重要性
  data.filter <- sbf(ttt3,ytt,sbfControl = sbfControl(functions=rfSBF, verbose=F, method='cv'))
  x <- ttt3[data.filter$optVariables]
  #储存选取变量
  Choose<-names(x)
  print(Choose)
}

