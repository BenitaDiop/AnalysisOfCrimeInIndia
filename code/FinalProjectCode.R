---
title: "Crime Against Women In India Analysis"
author: "Benita Diop"
output: 
  pdf_document: 
    latex_engine: 
      xelatex
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


```{r}
#### Data Preprocessing 
##### Loading d2 = Cases Under Crime Against Women #####
# d2 <- read_csv("https://raw.githubusercontent.com/BenitaDiop/9891FinalProject/main/datasets/Crime_In_India/42_Cases_under_crime_against_women.csv", col_names = T)

######  Loading d3 =Arrest Against Crime Against Women ######
# d3 <- read_csv("https://raw.githubusercontent.com/BenitaDiop/9891FinalProject/main/datasets/Crime_In_India/43_Arrests_under_crime_against_women.csv", col_names = T)

# dt2 <- merge(d2,d3, by= c("Area_Name","Year","Group_Name", "Sub_Group_Name"))

# dt2 <- fastDummies::dummy_cols(dt2, select_columns = "Group_Name")
# names(dt2)
# names(dt2)[35] <-"Cruelty_by_Husband_and_Relatives"
# names(dt2)[36] <-"DowryDeaths"
# names(dt2)[37] <-"Immoral_Traffic_Acts"
# names(dt2)[38] <-"ImportationOfGirls"
# names(dt2)[39] <-"Indecent_Representation_of_Women"
# names(dt2)[40] <-"Kidnapping&Abduction"
# names(dt2)[41] <-"Molestation"
# names(dt2)[42] <-"Rape"
# names(dt2)[43] <-"Sati_Prevention"
# names(dt2)[44] <-"SexualHarrasment"
# names(dt2)[45] <-"TotalCrimeAgainstWomen"
# data <- subset(dt2, select = -c(Area_Name, Year, Group_Name, Sub_Group_Name))
# write.csv(data, file = 'CrimeInIndia.csv')
# 
# 

```


```{r, message=FALSE}
library(readr)
library(plyr)
library(ggthemes)
library(glmnet)
library(randomForest)
library(dplyr)
library(ggplot2)
library(ISLR)
library(randomForest)
library(gridExtra)
```

```{r, warning=FALSE, message=FALSE}
dataset <- read_csv("https://raw.githubusercontent.com/BenitaDiop/AnalysisOfCrimeInIndia/main/datasets/CrimeInIndia.csv", col_names = T)
dataset <- subset(dataset, select = -c(X1))

```


Submit a proposal on the Discussion Board on Blackboard in which you: 
    
  (a) Describe the response variable and the predictors. How was the data collected?
  
```{r}
dataset %>% glimpse()
```  

  (b) Impute missing data-points with their mean. What is n and p?
```{r}
X <- as.matrix(subset(dataset, select =c(2:41)))
y <- as.matrix(subset(dataset ,select = -c(2:41)))
n        =    nrow(X)
p        =    ncol(X)
M              =     50   #total loops 
num_thrs       =     100 
n.train        =     floor(0.8*n)
n.validation  =     n-n.train
print(paste( n, "Number of Oberservations", "&", p, "Number of Features"))


```
    
  (c) Standardize the numerical predictors using equation (6.6) in the ISLR book.
    
        • The number of features p is at least 40.
        
        • The sample size n should be at least ten times the number of features p.
```{r}
require(stats)
data <- scale(dataset, center = FALSE, scale = apply(dataset, 2, sd, na.rm = TRUE))
```




```{r, message=FALSE, warning=FALSE, error=FALSE}

hist_y <-dataset %>%
  ggplot(aes(Cases_Acquitted_or_Discharged ))+
  geom_histogram(color = "darkblue", fill="lightblue",bins = 30)+
  geom_vline(aes(xintercept=mean(Cases_Acquitted_or_Discharged)), color="black",
             linetype="dashed")+
  labs(title="Histogram of Cases Aquitted or Discharged",x="Weight(kg)", y = "Count")+
  theme_classic()
hist_log_y <-dataset %>%
  ggplot(aes(log10(Cases_Acquitted_or_Discharged )))+
  geom_histogram(color = "darkblue", fill="lightblue",bins = 30)+
  geom_vline(aes(xintercept=mean(log10(Cases_Acquitted_or_Discharged))))+
  labs(title="Histogram of log10(Cases Aquitted or Discharged)",x="Weight(kg)", y = "Count")+
  theme_classic()

grid.arrange(hist_y, hist_log_y, nrow=2)
```







For each $n_{train}$ = 0.8n, repeat the following 100 times, do the following for the different models mentioned below.

    
  (a) Randomly split the dataset into two mutually exclusive datasets $D_{test}$ and $D_{train}$ with size ntest and ntrain such that $n_{train} + n_{test} = n$.
```{r}
ntrain = floor((0.8)*(n))
ntest = n - ntrain
train = sample.int(n, ntrain)
dtrain = data[train,]
dtest = data[-train,]

M              =     100
Rsq.test.rf    =     rep(0,M)  # rf= randomForest
Rsq.train.rf   =     rep(0,M)
Rsq.test.elas    =     rep(0,M)  #elas = elastic net
Rsq.train.elas   =     rep(0,M)
Rsq.test.rid    =     rep(0,M)  # rid=ridge
Rsq.train.rid   =     rep(0,M)
Rsq.test.las    =     rep(0,M)  #las=lasso
Rsq.train.las   =     rep(0,M)

Res.test.rf    =     rep(0,M)  # rf= randomForest
Res.train.rf   =     rep(0,M)
Res.test.elas    =     rep(0,M)  #elas = elastic net
Res.train.elas   =     rep(0,M)
Res.test.rid    =     rep(0,M)  # rid=ridge
Res.train.rid   =     rep(0,M)
Res.test.las    =     rep(0,M)  #las=lasso
Res.train.las   =     rep(0,M)

elas.time<-c(rep(0,M)) #empty vectors to store time
rid.time<-c(rep(0,M))
las.time<-c(rep(0,M))
rf.time<-c(rep(0,M))

elas.test.res<-c() # empty vectors for residual storage
rid.test.res<-c()
las.test.res<-c()
rf.test.res<-c()
elas.train.res<-c()
rid.train.res<-c()
las.train.res<-c()
rf.train.res<-c()

```
    
  (b) Use $D_{learn}$ to fit lasso, elastic-net α = 0.5, ridge, and random forrest.

  (c) Tune the $\lambda$s using 10-fold CV.

    
  (d) For each estimated model calculate
  
  $$R^2{test} = 1 - \frac{\frac{1}{n_{test}}\sum_{\substack{i \in D_{test}}}(y_i - \hat{y}_i)^2}{\frac{1}{n} \sum^n_{i=1}(y_i - \bar{y}^2)}$$


```{r, warning=FALSE, message=FALSE}
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  start.time<-Sys.time()
  elas.cv.fit      =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
  elas.fit         =     glmnet(X.train, y.train, alpha = 0.5, lambda = elas.cv.fit$lambda.min)
  end.time<-Sys.time()
  y.train.hat      =     predict(elas.fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(elas.fit, newx = X.test, type = "response") 
  Rsq.test.elas[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.elas[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  elas.time[m]<-elas.time+(end.time-start.time)
  elas.test.res<-y.test - y.test.hat
  elas.train.res<-y.train - y.train.hat
  
  # fit ridge and calculate and record the train and test R squares 
  start.time<-Sys.time()
  rid.cv.fit           =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  rid.fit              =     glmnet(X.train, y.train, alpha = 0, lambda = rid.cv.fit$lambda.min)
  end.time<-Sys.time()
  y.train.hat      =     predict(rid.fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(rid.fit, newx = X.test, type = "response")
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2) 
  rid.time[m]<-rid.time+(end.time-start.time)
  rid.test.res<-y.test - y.test.hat
  rid.train.res<-y.train - y.train.hat
  
  # fit lasso and calculate and record the train and test R squares 
  start.time<-Sys.time()
  las.cv.fit           =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  las.fit              =     glmnet(X.train, y.train, alpha = 1, lambda = las.cv.fit$lambda.min)
  end.time<-Sys.time()
  y.train.hat      =     predict(las.fit, newx = X.train, type = "response")
  y.test.hat       =     predict(las.fit, newx = X.test, type = "response") 
  Rsq.test.las[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.las[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2) 
  las.time[m]<-las.time+(end.time-start.time)
  las.test.res<-y.test - y.test.hat
  las.train.res<-y.train - y.train.hat
  
  # fit RF and calculate and record the train and test R squares 
  start.time<-Sys.time()
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  end.time<-Sys.time()
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  rf.time[m]<-rf.time+(end.time-start.time)
  rf.test.res<-y.test - y.test.hat
  rf.train.res<-y.train - y.train.hat
  
  cat( '\n',m,
       '\nRsq.test.rf', Rsq.test.rf[m], 'Rsq.train.rf', Rsq.train.rf[m],
       '\nRsq.test.elas', Rsq.test.elas[m],'Rsq.train.elas',Rsq.train.elas[m],
       '\nRsq.test.rid', Rsq.test.rid[m],'Rsq.train.rid',Rsq.train.rid[m],
       '\nRsq.test.las',Rsq.test.las[m],'Rsq.train.las',Rsq.train.las[m])
}
```


```{r}
#making R^2 dataframe
rf_test<-cbind(rep('random_forest'),rep('test'),Rsq.test.rf)
rf_train<-cbind(rep('random_forest'),rep('train'),Rsq.train.rf)
elas_test<-cbind(rep('elastic_net'),rep('test'),Rsq.test.elas)
elas_train<-cbind(rep('elastic_net'),rep('train'),Rsq.train.elas)
rid_test<-cbind(rep('ridge'),rep('test'),Rsq.test.rid)
rid_train<-cbind(rep('ridge'),rep('train'),Rsq.train.rid)
las_test<-cbind(rep('lasso'),rep('test'),Rsq.test.las)
las_train<-cbind(rep('lasso'),rep('train'),Rsq.train.las)

r2<-as.data.frame(rbind(rf_test,rf_train,elas_test,elas_train,rid_test,rid_train,las_test,las_train))
r2$V1<-as.factor(r2$V1)
r2$V2<-as.factor(r2$V2)
r2$Rsq.test.rf<-as.numeric(r2$Rsq.test.rf)
colnames(r2)<-c('model','set','rsq')

```


```{r}
#residual dataframe
rf_test_res<-cbind(rep('random_forest'),rep('test'),rf.test.res)
rf_train_res<-cbind(rep('random_forest'),rep('train'),rf.train.res)
elas_test_res<-cbind(rep('elastic_net'),rep('test'),elas.test.res)
elas_train_res<-cbind(rep('elastic_net'),rep('train'),elas.train.res)
rid_test_res<-cbind(rep('ridge'),rep('test'),rid.test.res)
rid_train_res<-cbind(rep('ridge'),rep('train'),rid.train.res)
las_test_res<-cbind(rep('lasso'),rep('test'),las.test.res)
las_train_res<-cbind(rep('lasso'),rep('train'),las.train.res)


res_data<-as.data.frame(rbind(rf_test_res,rf_train_res,elas_test_res,elas_train_res,rid_test_res,rid_train_res,
                             las_test_res,las_train_res))
glimpse(res_data)
colnames(res_data)<-c('model','set','residual')
res_data$residual<-as.numeric(res_data$residual)

```




```{r, warning=FALSE, message=FALSE}
#time datframe
rf_time<-cbind(rep('random_forest'),rf.time)
rid_time<-cbind(rep('ridge'),rid.time)
elas_time<-cbind(rep('elastic_net'),elas.time)
las_time<-cbind(rep('lasso'),las.time)

times<-as.data.frame(rbind(elas_time,rid_time,las_time,rf_time))
colnames(times)<-c('model','time')
times$time<-as.numeric(times$time)

times %>% group_by(model) %>% summarize(Minimum=min(time),Average=mean(time),Maximum=max(time))


```


```{r}
#R^2 plot
ggplot(r2,aes(x=model,y=rsq,fill=model))+geom_boxplot()+
  facet_wrap(~set)+theme_light()+
  ggtitle('')+ylab('R squared')+xlab('Model')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())+
  ggtitle('R^2 Comparison')

#residual plot
ggplot(res_data,aes(x=model,y=residual,fill=model))+geom_boxplot()+ 
  facet_wrap(~set)+theme_light()+
  ggtitle('')+ylab('Residuals')+xlab('Model')+
  theme(plot.title = element_text(hjust = 0.5))+theme()+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())+
  ggtitle('Residual comparison')


```

Create a presentation with less than 6 slides. Your objective is to be clear and concise. Hence I recommend the following:

  (a) a brief description of the nature of the data, shape, etc as discussed above. (1 slide)
```{r}

```

  (b) Show the side-by-side boxplots of $R^2{test}$, $R^2_{train}$. (1 slide)
```{r}
#making R^2 dataframe
rf_test<-cbind(rep('random_forest'),rep('test'),Rsq.test.rf)
rf_train<-cbind(rep('random_forest'),rep('train'),Rsq.train.rf)
elas_test<-cbind(rep('elastic_net'),rep('test'),Rsq.test.elas)
elas_train<-cbind(rep('elastic_net'),rep('train'),Rsq.train.elas)
rid_test<-cbind(rep('ridge'),rep('test'),Rsq.test.rid)
rid_train<-cbind(rep('ridge'),rep('train'),Rsq.train.rid)
las_test<-cbind(rep('lasso'),rep('test'),Rsq.test.las)
las_train<-cbind(rep('lasso'),rep('train'),Rsq.train.las)

r2<-as.data.frame(rbind(rf_test,rf_train,elas_test,elas_train,rid_test,rid_train,las_test,las_train))
r2$V1<-as.factor(r2$V1)
r2$V2<-as.factor(r2$V2)
r2$Rsq.test.rf<-as.numeric(r2$Rsq.test.rf)
colnames(r2)<-c('model','set','rsq')
```
  
  
  
  (c) For one on the 100 samples, create 10-fold CV curves for lasso, elastic-net $\alpha = 0.5$, ridge. (1 slide).
```{r}
#CV PLOTS
par(mfrow=c(3,1))
plot(las.cv.fit, sub = paste("Lasso:", las.cv.fit$lambda.min)) 
plot(elas.cv.fit, sub = paste("Elastic Net:", elas.cv.fit$lambda.min))
plot(rid.cv.fit, sub = paste("Ridge", rid.cv.fit$lambda.min))
par(mfrow=c(1,1))

fit.las<-glmnet(X, y,alpha=1)
fit.rid<-glmnet(X, y,alpha=0)

```

  (d) For one on the 100 samples, show the side-by-side boxplots of train and test
residuals (1 slide). Comment on the distribution and size of the residuals.
```{r}

```

  (e) Present bar-plots (with bootstrapped error bars) of the estimated coefficients, and the importance of the parameters. If you have something interesting to say about coefficients that are (or are not important) say it. (1 slide)
```{r}
#COEFFICIENTS PLOTS
plot(fit.las,xvar="lambda",label=TRUE)
plot(fit.rid,xvar="lambda",label=TRUE)
```



```{r}
bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.elas.bs     =     matrix(0, nrow = p, ncol = bootstrapSamples)   
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.las.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples) 

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs elas
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit$lambda.min)  
  beta.elas.bs[,m]   =     as.vector(fit$beta)
  
  # fit bs las
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit$lambda.min)  
  beta.las.bs[,m]   =     as.vector(fit$beta)
  
  # fit bs rid
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m]   =     as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
elas.bs.sd  = apply(beta.elas.bs, 1, "sd")
las.bs.sd   = apply(beta.las.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")


```



```{r}
# fit rf to the whole data
rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)

# fit elas to the whole data
cv.fit           =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
elas.fit              =     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit$lambda.min)

#fit las to the whole data
las.cv.fit           =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
las.fit              =     glmnet(X.train, y.train, alpha = 1, lambda = las.cv.fit$lambda.min)

#fit rid to the whole data
rid.cv.fit           =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
rid.fit              =     glmnet(X.train, y.train, alpha = 0, lambda = rid.cv.fit$lambda.min)


betaS.rf               =     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.elas               =     data.frame(names(X[1,]), as.vector(elas.fit$beta), 2*elas.bs.sd)
colnames(betaS.elas)     =     c( "feature", "value", "err")

betaS.las               =     data.frame(names(X[1,]), as.vector(las.fit$beta), 2*las.bs.sd)
colnames(betaS.las)     =     c( "feature", "value", "err")

betaS.rid               =     data.frame(names(X[1,]), as.vector(rid.fit$beta), 2*rid.bs.sd)
colnames(betaS.rid)     =     c( "feature", "value", "err")
```

  (f) Summary slide: summarize the performance and the time need to train each
model in a table and comment on it. (1 slide)
```{r}
y <- as.vector(y)
# fit rf to the whole data
rf  = randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
elastic.cv.fit = cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
elastic.fit = glmnet(X, y, alpha = 0.5, lambda = elastic.cv.fit$lambda.min)

# fit la to the whole data
lasso.cv = cv.glmnet(X, y, alpha = 1)
lasso.fit = glmnet(X, y, alpha = 0.5, lambda = lasso.cv$lambda.min)
# fit ridge to the whole data
ridge.cv = cv.glmnet(X, y, alpha = 0)
ridge.fit = glmnet(X, y, alpha = 0.5, lambda = ridge.cv$lambda.min)

cols = 1:length(names(X[1,])) %>% as.factor()

```


```{r}

#calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
la.bs.sd = apply(beta.las.bs, 1, "sd")
ri.bs.sd = apply(beta.rid.bs, 1, "sd")
rf.bs.sd = apply(beta.rf.bs, 1, "sd")
en.bs.sd = apply(beta.elas.bs, 1, "sd")

betaS.lasso = data.frame(feature = cols, 
                          value = as.vector(lasso.fit$beta), 
                          error = 2*la.bs.sd, 
                          model = "elastic")
betaS.ridge = data.frame(feature = cols, 
                          value = as.vector(ridge.fit$beta), 
                          error = 2*ri.bs.sd, 
                          model = "elastic")
betaS.rf = data.frame(feature = cols, 
                       value = as.vector(rf$importance[,1]), 
                       error = 2*rf.bs.sd, 
                       model = "elastic")
betaS.elastic = data.frame(feature = cols, 
                          value = as.vector(elastic.fit$beta), 
                          error = 2*en.bs.sd, 
                          model = "elastic")
```






```{r}
rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value))+
  geom_bar(stat = "identity",fill="white", colour="black")+
  ylab('RF')+
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

elasPlot =  ggplot(betaS.elas, aes(x=feature, y=value))+
  geom_bar(stat = "identity", fill="white", colour="black")    + ylab('EN')+
  geom_errorbar(aes(ymin=value-err,
                    ymax=value+err), width=.2)

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +ylab('LAS')+
  geom_errorbar(aes(ymin=value-err,
                    ymax=value+err), width=.2)

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +ylab('RID')+
  geom_errorbar(aes(ymin=value-err,
                    ymax=value+err), width=.2)


grid.arrange(rfPlot,elasPlot,lasPlot,ridPlot, nrow = 4)
```

```{r}
# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.elas$feature   =  factor(betaS.elas$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.las$feature    =  factor(betaS.las$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +ylab('RF')+
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())

elasPlot =  ggplot(betaS.elas, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    + ylab('EN')+
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +ylab('LAS')+
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +ylab('RID')+
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  theme(axis.title.x=element_blank(),axis.text.x=element_blank())


grid.arrange(rfPlot,elasPlot,lasPlot,ridPlot, nrow = 4)



```











```{r, include=FALSE}
# . Submission guidelines:
# 
# • Create an account on https://github.com/ and upload your code, data and the
# pdf of your presentation. The code and data should be such that if I download
# it in one folder, I should be able to run it, and create the figures you present in
# your pdf. Reproducibility is an important step in any data analysis project.

# • Upload your 5 minute video on vimeo (with or without a password, depending
# on what you are more comfortable). One way to create the video is to full-screen
# your pdf presentation, and talk over it while Quicktime is recording your screen
# and sound.

# • Submit a SINGLE PDF with hyperlinks that take me to your github page, and
# the vimeo page where your presentation is uploaded. See the template.
```








