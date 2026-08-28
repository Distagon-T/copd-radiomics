data1<-read.csv(file='/media/luviagelita/ubuntu/aLIEz/bob_56_2d.csv',header=T)
data2<-read.csv(file='/media/luviagelita/ubuntu/aLIEz/guangdong_135_2d.csv',header=T)
data3<-read.csv(file='/media/luviagelita/ubuntu/aLIEz/henan_2d.csv',header=T)
data4<-read.csv(file='/media/luviagelita/ubuntu/aLIEz/NC_273_2d.csv',header=T)

p_value<-matrix(data=NA,nrow=7,ncol=507,byrow=T)
wilcox_result <- c()
kruskal_result <- c()

for (i in 1:507){

  A <- as.numeric(data1[,i])
  B <- as.numeric(data2[,i])
  C <- as.numeric(data3[,i])
  D <- as.numeric(data4[,i])
  
  p_value[1,i] <-wilcox.test(A,B)$p.value
  p_value[2,i] <-wilcox.test(A,C)$p.value
  p_value[3,i] <-wilcox.test(A,D)$p.value
  p_value[4,i] <-wilcox.test(B,C)$p.value
  p_value[5,i] <-wilcox.test(B,D)$p.value
  p_value[6,i] <-wilcox.test(C,D)$p.value
  if (p_value[1,i] > 0.05 &&
      p_value[2,i] > 0.05 &&
      p_value[3,i] > 0.05 &&
      p_value[4,i] > 0.05 &&
      p_value[5,i] > 0.05 &&
      p_value[6,i] > 0.05) wilcox_result <- c(wilcox_result,i)
  
  p_value[7,i] <- kruskal.test(list(g1=A,g2=B,g3=C,g4=D))$p.value
  
  if (p_value[7,i]>0.05) kruskal_result <- c(kruskal_result,i)

  if (p_value[7,i]>0.05) print(i)
}



#which(p_value>0.05)