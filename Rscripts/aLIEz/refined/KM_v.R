library(rms)

inputData <- read.csv(file="/media/luviagelita/ubuntu/R/aLIEz/refined/guangdong_3d_refined.csv",header=T)
inputData <- as.data.frame(inputData)
status <- as.numeric(inputData[,10])
time <- as.numeric(inputData[,11])

radiomics_score <- vector(length=nrow(inputData))
for (i in 1:nrow(inputData))
{ if (radiomics[i] > 0 ) radiomics_score[i] <- "indicator +"
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
         ## conf.int=.95,                       # show 95% CI,
         conf='none',                           # change type of CI
         label.curves = list(keys = "lines"),   # legend instead of direct label
         levels.only  = TRUE,                   # show only levels, no label
         ## col=c('red','black','blue'),           # change legend color
         ## fun = function(x) {1 - x},          # Cumulative probability plot
         ##loglog   = TRUE,                    # log(-log Survival) plot
         ## logt     = TRUE,                    # log time
         ## time.inc = 5,                          # time increment
         ## dots     = TRUE,                    # dot grid
         ## n.risk   = TRUE,                       # show number at risk
         ## y.n.risk = 0.01,                       # Change position of number at risk
         ##cex.n.risk = 0.6,                       # change character size for number at risk
         stitle='Radiomics')



data.survdiff <- survdiff(Surv(time, status) ~ radiomics_score, data = inputData)
p.val = 1 - pchisq(data.survdiff$chisq, length(data.survdiff$n) - 1)
HR = (data.survdiff$obs[2]/data.survdiff$exp[2])/(data.survdiff$obs[1]/data.survdiff$exp[1])
up95 = exp(log(HR) + qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
low95 = exp(log(HR) - qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
print(data.survdiff)

cox <-coxph(Surv(time, status) ~ radiomics, data=inputData)
summary(cox)
