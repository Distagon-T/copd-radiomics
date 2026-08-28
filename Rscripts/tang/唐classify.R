library(Hmisc); library(grid); library(lattice);library(Formula); library(ggplot2) 
library(rms);
source("val.prob.ci.dec08.r");
source("HLtest.r")
source("dca.r")
pre.train<-spss.get("pretrain.sav")
pre.test<-spss.get("pretest.sav")
dd=datadist(pre.train)
options(datadist="dd")

#rms 包 里的lrm回归
f.train <- lrm(label~radiomics.signature + CEA + CA199, data=pre.train, x=T, y=T,linear.predictors=T);
nom <- nomogram(f.train, fun= function(x)1/(1+exp(-x)), # or fun=plogis
                lp=F, funlabel="Probability of pCR ")
plot(nom)

rcorrcens(label ~ predict(f.train), data =  pre.train);
val.prob.ci(logit=f.train$linear.predictors,y=f.train$y, pl=T,smooth=F,logistic.cal=F, g=3,connect.group =T,
            xlab="Predicted risk",
            ylab="Observed risk",riskdist='predicted', dist.label=-0.95, cutoff=.2)

#HL检验 >0.05代表可以用
hl.ext2(p=plogis(f.train$linear.predictors),y=f.train$y,g=3,df=2)

# f train为得到的模型
test.pred  <- predict(object=f.train, newdata = pre.test) 
val.prob.ci(logit=test.pred,y=pre.test$label, pl=T,smooth=F,logistic.cal=F, g=3,connect.group =T,
            xlab="Predicted risk",
            ylab="Observed risk",riskdist='predicted', dist.label=-0.95, cutoff=.2)
hl.ext2(p=plogis(test.pred),y=pre.test$label,g=3,df=2)

#画决策曲线, yvar为label xmatrix为模型输出
dca.train   <- dca(yvar=f.train$y, xmatrix=plogis(f.train$linear.predictors), prob="Y") # tumor
dca.test   <- dca(yvar=pre.test$label, xmatrix=plogis(test.pred), prob="Y") # tumor


