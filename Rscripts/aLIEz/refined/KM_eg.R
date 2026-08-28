library(stats)
library(survival)

## Information of data

data(package = "survival")  # List datasets in survival package
help(bladder1)              # Description of data
head(bladder1)              # Show first 6 rows
str(bladder1)               # Check type of variables
summary(bladder1)           # Statistical summary

## Get the final data with nonzero follow-up
bladder1$time <- 
  as.numeric(bladder1$stop -
               bladder1$start)
summary(bladder1$time)
bladder1 <- subset(bladder1,status<=1 & time>0)


## Step1 Create Kaplan-Meier curve and estimate median survial/event time 
## The "log-log" confidence interval is preferred.
## Create overval Kaplan-Meier curve
km.as.one <- survfit(Surv(time, status) ~ 1, data = bladder1, 
                     conf.type = "log-log")

## Create Kaplan-Meier curve stratified by treatment
km.by.trt <- survfit(Surv(time, status) ~ treatment, data = bladder1, 
                     conf.type = "log-log")

## Show simple statistics of Kaplan-Meier curve
km.as.one
km.by.trt

## See survival estimates at given time (lots of outputs)
summary(km.as.one)
summary(km.by.trt)

## Plot Kaplan-Meier curve without any specification
plot(km.as.one)
plot(km.by.trt)

## Plot Kaplan-Meier curve Without confidence interval and mark of event
plot(km.as.one, conf = F, mark.time = F)
plot(km.by.trt, conf = F, mark.time = F)


## step2 Create a simple cox regression and estimate HR:
model1 <-coxph(Surv(time, status) ~ treatment + number +size, 
               data=bladder1)

## Model output
summary(model1)                    # Output summary information
confint(model1)                    # Output 95% CI for the coefficients
exp(coef(model1))                  # Output HR (exponentiated coefficients)
exp(confint(model1))               # 95% CI for exponentiated coefficients
predict(model1, type="risk")       # predicted values
residuals(model1, type="deviance") # residuals


## Step3 Check for violation of proportional hazard (constant HR over time)
model1.zph <- cox.zph(model1)
model1.zph                    
## Note: p value of treatmentthiotepa <0.05
## GLOBAL p value is more important than p values of treatmentthiotepa 
## Only GLOBAL p value <0.05 is neccesary for solution of violation
## Following steps are just for providing examples

## Displays a graph of the scaled Schoenfeld residuals, along with a smooth curve.
plot(model1.zph)

## Step4 Solution of violating of proportional hazard

## Solution 1 Stratify: create a stratified cox model,namely "contitional cox model "
model2 <-coxph(Surv(start, stop, status) ~ number +size + strata (treatment), 
               data=bladder1)
summary(model2) 
model2.zph <- cox.zph(model2)
model2.zph

## Solution 1 Create a simple time-dependent model without time-dependent variable
model3 <-coxph(Surv(start, stop, status) ~ number +size + treatment, 
               data=bladder1)
summary(model3) 

## Create a time-dependent model with time-dependent variable
bladder1$thiotepa <- 
  as.numeric (bladder1$treatment=="thiotepa")
model4 <-coxph(Surv(start, stop, status) ~ number +size + treatment
               + thiotepa:time,     ## Note: "thiotepa:time" but not "thiotepa*time"
               data=bladder1)
summary(model4) 

## Step5 Performs stepwise model selection by AIC
model5 <-step(model4)  ## Default direction is "backward"
summary(model5)


## rms::survplot() function
## Load rms package
library(rms)

## Plot Kaplan-Meier curve without any specification
fit1 <- npsurv(Surv(time, status) ~ 1, data = bladder1)
fit2 <- npsurv(Surv(time, status) ~ treatment, data = bladder1)
survplot(fit1)
survplot(fit2)

## Plot Kaplan-Meier curve Without confidence interval
survplot(fit1, conf = "none")
survplot(fit2, conf = "none")

## More options for Kaplan-Meier curve
survplot(fit2, 
         xlab = "Time, month",                  # add x-axis label
         ylab = "Survival probability, %",      # add y-axis label
         ## xlim=c(0,60),                       # add x-axis limits
         ## ylim=c(-0.1,1),                     # add y-axis limits
         ## conf.int=.95,                       # show 95% CI,
         conf='none',                           # change type of CI
         label.curves = list(keys = "lines"),   # legend instead of direct label
         levels.only  = TRUE,                   # show only levels, no label
         col=c('red','green','blue'),           # change legend color
         ## fun = function(x) {1 - x},          # Cumulative probability plot
         ## loglog   = TRUE,                    # log(-log Survival) plot
         ## logt     = TRUE,                    # log time
         time.inc = 5,                          # time increment
         ## dots     = TRUE,                    # dot grid
         n.risk   = TRUE,                       # show number at risk
         y.n.risk = 0.01,                       # Change position of number at risk
         cex.n.risk = 0.6                       # change character size for number at risk
)