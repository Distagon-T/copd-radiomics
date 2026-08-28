# Program for testicular cancer analyses: performance evaluation
# Ewout Steyerberg, Sept 2009
library(Hmisc)
library(rms)
library(foreign)

source('f:/Book/Cleveland/Www/HLtest.r')
source('f:/Book/Cleveland/Www/val.prob.ci.dec08.r')
# source('f:/Book/Cleveland/Www/dcaNov08.r')
source('f:/Book/Cleveland/www/dca.r')

#######################
# Import testis data  #
t821  <- read.spss('f:/Book/SPSS/t821.sav',use.value.labels=F, to.data.frame=T)
t821$tum  <- 1 - t821$nec
t821[1,]
#   patkey hosp study orchyr histr3 ter preafp prehcg     lnldhst   sqpost
#1     94    1     1     79      2   0      0      1 -0.09553243 5.477226
#  reduc10 nec matter tum
# 1       7   0      1   1

describe(t821)

n544  <- t821[t821$study==1,] # development set, n=544, JCO 1995
val  <- t821[t821$study==3,]  # Indiana validation set. n=273, J Urol 2001; no LDH values
valother  <- t821[t821$study==2 | t821$study==4,] # Val 172 JCO 1997; n=105 BrJC 2003; total n=277

#######################
# Fit a model in n544; 5 predictors
full <- lrm(tum ~ ter+preafp+prehcg+sqpost+reduc10,
			data=n544,x=T,y=T,linear.predictors=T)
full # c = 0.818

			       
# Brier max
B     <- mean((full$y) * (1-plogis(full$linear.predictors))^2 + 
          (1-full$y) * plogis(full$linear.predictors)^2)
B
Bmax  <- mean(full$y) * (1-mean(full$y))
Bmax
Bscaled <- 1 - B/Bmax
Bscaled

# Compare to Pearson R2
cor(x=plogis(full$linear.predictors), y=full$y)^2


#############################
# Externally validate the model 
lp.val  <- predict(object=full, newdata = val)

# Brier max
B     <- mean((val$tum) * (1-plogis(lp.val))^2 + 
          (1-val$tum) * plogis(lp.val)^2)
B
Bmax  <- mean(val$tum) * (1-mean(val$tum))
Bmax
Bscaled <- 1 - B/Bmax
Bscaled
cor(x=plogis(lp.val), y=val$tum)^2

## H-L tests
hl.ext2(p=plogis(full$linear.predictor),y=full$y,g=10,df=8)
hl.ext2(p=plogis(lp.val),y=val$tum,g=10,df=9)

## CI around c stat
cstatNo <- rcorr.cens(full$linear.predictors, full$y) 
cat(cstatNo[1], "[", cstatNo[1]-1.96/2*cstatNo[3], " - ", cstatNo[1]+1.96/2*cstatNo[3],"]")
cstatvAL <- rcorr.cens(lp.val, val$tum) 
cat(cstatvAL[1], "[", cstatvAL[1]-1.96/2*cstatvAL[3], " - ", cstatvAL[1]+1.96/2*cstatvAL[3],"]")

#############################
## Validation plots
par(mfrow = c(1,2), pty='s', mar=c(4.2,4,4,1),cex=0.95, font =1, col=1 )
# Apparent smooth设置成F就是线段
val.prob.ci(logit=full$linear.predictor,y=full$y, pl=T,smooth=T,logistic.cal=F, g=10,
			       xlab="Predicted risk without LDH",
			       ylab="Observed masses with tumor",riskdist='predicted',
             d1lab="Tumor", d0lab="Necrosis", dist.label=-0.95, cutoff=.2)
title("Development, n=544")

# External
val.prob.ci(logit=lp.val,y=val$tum, pl=T,smooth=T,logistic.cal=F, g=10,
			       xlab="Predicted risk without LDH",
			       ylab="Observed masses with tumor at validation",riskdist='predicted',
             d1lab="Tumor", d0lab="Necrosis", dist.label=-0.95, cutoff=.2)
title("Validation, n=273")


#############################
## Boxplots
par(mfrow = c(1,2), pty='s', mar=c(4.2,4,4,1),cex=0.95, font =1, col=1 )
boxplot(plogis(full$linear.predictors)~full$y,
    ylab="Predicted risk without LDH", xlab="Tumor",ylim=c(0,1))
boxplot(c(mean(plogis(full$linear.predictors[full$y==0])), mean(plogis(full$linear.predictors[full$y==1])))~ c(0,1),add=T,
  boxlty=0, staplelty=0, medlty=0, medlwd=0, medpch=15)
title(paste("Development: Slope=", 
  round(mean(plogis(full$linear.predictors[full$y==1])) - 
        mean(plogis(full$linear.predictors[full$y==0])),2),sep=""))

boxplot(plogis(lp.val)~val$tum,
    ylab="Predicted risk without LDH", xlab="Tumor at validation",ylim=c(0,1))
boxplot(c(mean(plogis(lp.val[val$tum==0])), mean(plogis(lp.val[val$tum==1])))~ c(0,1),add=T,
  boxlty=0, staplelty=0, medlty=0, medlwd=0, medpch=15)
title(paste("Validation: Slope=", 
  round(mean(plogis(lp.val[val$tum==1])) - 
        mean(plogis(lp.val[val$tum==0])),2),sep=""))


###########################
## Start decision curves ##
# dca(yvar, xmatrix, xstart=0.01, xstop=0.99, xby=0.01, prob)
# dca.dev <- dca(yvar=full$y, xmatrix=plogis(full$linear.predictor), prob="Y")  # necrosis
dcaNo   <- dca(yvar=full$y, xmatrix=plogis(full$linear.predictor), prob="Y") # tumor
dcaVal  <- dca(yvar=val$tum, xmatrix=plogis(lp.val), prob="Y") # tumor

# Net benefit at 20%,30%, 40% thresholds
dcaNo[c(20,30,40),1]-dcaNo[c(20,30,40),2]
dcaVal[c(20,30,40),1]-dcaVal[c(20,30,40),2]

par(mfrow = c(1,2), pty='s', mar=c(4.2,4,4,1),cex=1)
plot(x=dcaNo$threshold, y=dcaNo[,1], type='l', lty=1, lwd=2, las=1, ylim=c(-.05,.8),
     ylab="Net Benefit", xlab="Threshold risk for resection of tumor (%)", cex.lab=1.2)
lines(x=dcaNo$threshold, y=dcaNo[,2], lty=3, lwd=1)
lines(x=dcaNo$threshold, y=dcaNo[,3], lty=1, lwd=1)
arrows(20,max(dcaNo[,2],na.rm=T)-.05,20,0)

# External validation
plot(x=dcaVal$threshold, y=dcaVal[,1], type='l', lty=1, lwd=2, las=1, ylim=c(-.05,.8),
     ylab="Net Benefit at validation", xlab="Threshold risk for resection of tumor (%)", cex.lab=1.2)
lines(x=dcaVal$threshold, y=dcaVal[,2], lty=3, lwd=1)
lines(x=dcaVal$threshold, y=dcaVal[,3], lty=1, lwd=1)

arrows(20,max(dcaVal[,2],na.rm=T)-.05,20,0)
text(x=20,y=max(dcaVal[,2],na.rm=T), "Threshold")
text(x=80,y=.03, "Resection in none")
text(x=60,y=.2, "Resection\nin all")
text(x=75,y=.5, "Resection\nif tumor risk\n> threshold")
title("Validation, n=273")

## End decision curves   ##
###########################



########################
# Internal validation  #
val.full  <- validate(full, B=200)
val.full
val.full[1,1:5]/2 + .5  # index.corrected c stat 0.811

# Bootstrap discrimination slope and DCA
nrowB	<- nrow(n544)  # nrow from development set
B <- 20            # 200 bootstraps
matB <- matrix(NA,nrow=B,ncol=3) # Matrix for results
dimnames(matB) <- list(c(1:B), Cs(Slopeapp, Slopetest, optimism ))
matDCA  <- matrix(0,nrow=99,ncol=3) # Matrix for results of DCA
dimnames(matDCA) <- list(c(1:99), Cs(NBorig, NBval, NBoptimism ))

# Start loop
for (i in 1:B) {
if (i%%10==0) cat("Start Bootstrap sample nr", i, "\n")
Brows <- sample(nrowB,replace=T)

# Bsample is bootstrap sample from development set
Bsample	<- n544[Brows,]
devfull <- lrm(tum ~ ter+preafp+prehcg+sqpost+reduc10, data=Bsample,linear.predictors=T, x=T, y=T)
matB[i,1] <- mean(plogis(devfull$linear.predictors[devfull$y==1])) - 
            mean(plogis(devfull$linear.predictors[devfull$y==0]))
lp  <- full$x %*% devfull$coef[2:length(devfull$coef)] + devfull$coef[1] # lp with coefs from bootstrap
matB[i,2] <- mean(plogis(lp[full$y==1])) - mean(plogis(lp[full$y==0]))  # Testing on original sample

dcaorig   <- dca(yvar=devfull$y, xmatrix=plogis(devfull$linear.predictor), prob="Y") # bootstrap sample
dcaval    <- dca(yvar=full$y, xmatrix=plogis(devfull$linear.predictor), prob="Y") # Testing on original sample
matDCA[,1]  <- (i-1)/i * matDCA[,1] + 1/i * (dcaorig[,1] - dcaorig[,2])  # NB orig
matDCA[,2]  <- (i-1)/i * matDCA[,2] + 1/i * (dcaval[,1]  - dcaval[,2])   # NB validated

} # End for loop
cat("\n\nEnd! \n\n")
matB[,3] <- matB[,1] - matB[,2] # optimism per bootstrap
matDCA[,3] <- matDCA[,1] - matDCA[,2] # optimism per threshold
# matB and matDCA results
apply(matB,2,mean)
matDCA  # optimism estimates per threshold
dcaNo # original evaluation, apparent validation

# Make a shrunk model
full.shrunk <- full
full.shrunk$coef <- val.full[4,5] * full.shrunk$coef  
# val.full[4,5] is shrinkage factor; heuristic estimate (LR - df) / LR = (211-6)/211=0.97
# Estimate new intercept, with shrunk lp as offset variable, i.e. coef fixed at unity
full.shrunk$coef[1] <- lrm.fit(y=full$y, offset= full$x %*% full.shrunk$coef[2:7])$coef[1]
full.shrunk$coef /full$coef
full.shrunk

# Make a penalized model
p	<- pentrace(full, c(0,1,2,3,4,5,6,7,8,10,12,14,20))
plot(p, which='aic.c', xlim=c(0,15))
p$penalty         # Optimal penalty factor is 4
full.pen	<- update(full, penalty=p$penalty) 
full.pen$coef / full$coef 
full.pen

### End example performance of prediction models ###

