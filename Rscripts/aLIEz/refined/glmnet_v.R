rm(list=ls(all=TRUE))
####  ####
inputData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/Cross_Validation/2d_train_L.csv",header=T)

x <- data.frame(inputData[,1:9])
y <- as.factor(inputData[,10])

library(glmnet)
#glm.fit=glm(y ~.,data=x,family=binomial(link="logit"))
#for (a in seq(1,100,5))
#{
  lasso<-cv.glmnet(as.matrix(x),y,type.measure="class" ,nfolds = 10,family="binomial",alpha=0.95)
  radio<-predict(lasso,as.matrix(x))
  roc.lasso <- roc(y,as.numeric(radio))
  plot(roc.lasso)
  
  ##### test #######
  
  testData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/guangdong_2d_refined_L.csv",header=T)
  
  xt <- data.frame(testData[,1:9])
  yt <- as.vector(testData[,10])
  
  pred=predict(lasso,as.matrix(xt))#用模型对测试数据进行预测
  
  
  library(pROC)
  
  ROCt<-roc(yt,pred)
 plot(ROCt)
  
  
  ###### Survival Analysis ######
  library(rms)
  ##### ######
  inputData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/guangdong_2d_refined.csv",header=T)
  inputData <- as.data.frame(inputData)
  status <- as.numeric(inputData[,10])
  time <- as.numeric(inputData[,11])
  
  #temp = 0
  #n <- nrow(testData)
  #rs <- c()
  #for (i in 1:n) {
  #  sig <- 0
  #  for(j in 1:9){
  #    temp <- unname(glm.fit$coefficients[j])*inputData[i,j]
  #    sig <- sig + temp
  #  }
  #  rs <- c(rs,sig)
  #}
  
  
  
  radiomics_score <- vector(length=nrow(inputData))
  for (i in 1:nrow(inputData))
  { if (pred[i] > -0.05)  radiomics_score[i] <- "indicator +"
  else
    radiomics_score[i] <- "indicator -"
  }
  
  s <- Surv(time,status)
  
  Kaplan_Meier <- npsurv(Surv(time, status) ~ radiomics_score, data = inputData)
  survplot(Kaplan_Meier,
           xlab = "Time (days)",                  # add x-axis label
           ylab = "Survival",      # add y-axis label
           xlim=c(0,2000),                      # add x-axis limits
           ylim=c(0,1),                     # add y-axis limits
           time.inc=500,
           conf.int=.95,                       # show 95% CI,
           ## conf='none',                           # change type of CI
           label.curves = list(keys = "lines"),   # legend instead of direct label
           levels.only  = TRUE,                   # show only levels, no label
           ## col=c('red','black','blue'),           # change legend color
           ## fun = function(x) {1 - x},          # Cumulative probability plot
           ##loglog   = TRUE,                    # log(-log Survival) plot
           ## logt     = TRUE,                    # log time
           ## time.inc = 5,                          # time increment
           ## dots     = TRUE,                    # dot grid
           n.risk   = TRUE,                       # show number at risk
            y.n.risk = 0.01,                       # Change position of number at risk
           ##cex.n.risk = 0.6,                       # change character size for number at risk
           stitle='Radiomics')
  
  data.survdiff <- survdiff(Surv(time, status) ~ radiomics_score, data = inputData,rho=0)
  p.val = 1 - pchisq(data.survdiff$chisq, length(data.survdiff$n) - 1)
  HR = (data.survdiff$obs[2]/data.survdiff$exp[2])/(data.survdiff$obs[1]/data.survdiff$exp[1])
  up95 = exp(log(HR) + qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
  low95 = exp(log(HR) - qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
  print(data.survdiff)
  
  cox <-coxph(Surv(time, status) ~ pred, data=inputData)
  summary(cox) 
 
#}