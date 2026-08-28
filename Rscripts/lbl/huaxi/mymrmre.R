mymrmre<-function(x,y,f,s=1){
mdata<-cbind(x,y)
mdata<-as.data.frame(mdata)
dd <- mRMR.data(data = mdata) 
ensemble<-mRMR.ensemble(data = dd, target_indices = c(ncol(mdata)),  solution_count = s, feature_count = f)
  t<-scores(ensemble)
  sol<-solutions(ensemble)
  sol<-data.frame(sol)
  sol<-as.matrix(sol)
  Choose<-names(traindata)
  Choose2<-Choose[sol]
return(list(score=t,Choose=Choose2))
}