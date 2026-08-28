library(Hmisc)
library(survAUC)

data<-read.csv(file.choose())
features<-as.data.frame(data[,1:((ncol)(data)-2)])
s<-as.numeric(data[,(ncol)(data)-1])
t<-as.numeric(data[,(ncol(data))])
y<-Surv(t,s)
c_i <- c()
picked_features_CI <- c()
for (i in 1:507) {

coxmodel <- coxph(y~ features[,i]  ,data=features)
sum.cox <- summary(coxmodel)
c_index <- sum.cox$concordance 
c_i <- c(c_i,c_index[1])

if (i %in% kruskal_result_3d) 
  picked_features_CI <- c(picked_features_CI,c_index[1])

}

#print(max(c_i))

