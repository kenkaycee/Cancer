library(tidyverse);library(caret) # Import Libraries
cancer<- read.csv(file.choose(), stringsAsFactors = F) # Load data
str(cancer) # shows structure of cancer data 
summary(cancer)# summary statistics of variables in cancer dataset
names(cancer) # Column names
## remove id column as not needed for modelling
cancer<- cancer[-1]
cancer$diagnosis # outcome variable
##  convert diagnosis variables to factors 
cancer$diagnosis<-factor(cancer$diagnosis, levels = c("B", "M"),
                         labels = c("Benign","Malignat"))
table(cancer$diagnosis) %>% prop.table() # 63% is benign and 37% malignant 

## splits cancer into train and test dataset
set.seed(100) ## ensures reproducibilit

trainindex<- createDataPartition(cancer$diagnosis, p=0.75, list = F)
train_cancer<- cancer[trainindex,]
test_cancer<- cancer[-trainindex,]
## compare frequencies of diagnosis in test and train data set against the original canceer dataset
train_prop<-round(prop.table(table(train_cancer$diagnosis))*100,1)
test_prop<- round(prop.table(table(test_cancer$diagnosis))*100,1)
original_prop<- round(prop.table(table(cancer$diagnosis))*100,1) 
freq<- data.frame(cbind(original_prop,train_prop,test_prop))
colnames(freq)<- c("Original","Training", "Testing")
freq ## the frequencies are approximately similar

##  parameter tuning 

fitCtrl<- trainControl(method = "repeatedcv", number = 10, repeats = 3)## repeated k_fold CV
## fit knn classification using caret
set.seed(100)
knnFit<- train(diagnosis~.,data = train_cancer, method="knn", trControl=fitCtrl, tuneLength=20, 
               preProcess=c("center","scale"), metric="Accuracy")
knnFit ## shows that best model was K=7 on the train data with accuracy of 97%
##  plot the accuracy rate for different neighbours
plot(knnFit)
knnFit %>% ggplot()+
  scale_x_continuous(breaks = c(1:43))+
  theme_bw()##  shows K= 7 has highest accuracy

##  evaluate model perfomance using test data
knnPredict<- predict(knnFit, test_cancer)
cmatKnn<- confusionMatrix(knnPredict, test_cancer$diagnosis, positive = "Malignat")
cmatKnn # Accuracy rate of 96%

##  predicing cancer diagnosis using Logistic Regression
set.seed(100)
logFit<- train(diagnosis~., data = train_cancer, method="glm", family="binomial", metric="Accuracy", tuneLength=20,
               trControl=fitCtrl)
logPredict<- predict(logFit, test_cancer)
cmatLog<- confusionMatrix(logPredict, test_cancer$diagnosis, positive = "Malignat")
cmatLog ## Accuracy of 94%

##  Prediciting cancer diagnosis using decision tree
set.seed(100)
rpartFit<- train(diagnosis~., data = train_cancer, method="rpart",  metric="Accuracy",
                 trControl=fitCtrl, tuneLength=20)
plot(rpartFit)
## plot thr tree
rpart.plot::rpart.plot(rpartFit$finalModel)
rpartPredict<- predict(rpartFit, test_cancer)
cmatRpart<- confusionMatrix(rpartPredict, test_cancer$diagnosis, positive = "Malignat")
cmatRpart ## Accuracy of 91%

##  predicting using lda
set.seed(100)
ldaFit<- train(diagnosis~., data = train_cancer, method="lda", metric="Accuracy", trControl=fitCtrl,tuneLength=20)
ldaFit
ldaPredict<- predict(ldaFit, test_cancer)
cmatLda<- confusionMatrix(ldaPredict, test_cancer$diagnosis, positive = "Malignat")
cmatLda ## Accuracy of 94%

##  predictinc cancer diagnoisi using QDA
qdaFit<- train(diagnosis~., data = train_cancer, method="qda", trControl=fitCtrl, metric="Accuracy", tuneLength=20)
qdaFit
qdaPredict<- predict(qdaFit, test_cancer)
cmatQda<- confusionMatrix(qdaPredict, test_cancer$diagnosis, positive = "Malignat")
cmatQda ##  Accuracy of 96%

##  Predicting Cancer diagnosis using Randomforest 
set.seed(100)
rfFit<- train(diagnosis~., data = train_cancer, method="rf", metric="Accuracy", 
              trControl=fitCtrl, tuneLength=20, importance=TRUE)
rfFit$finalModel %>% randomForest::importance()
varImp(rfFit) ##  shows the importance of the predictors in explaining the model 
varImp(rfFit) %>% ggplot()+geom_col(fill="red") ## visualize importance of predictors
varImp(rfFit) %>% plot(col="red")
rfFit %>% ggplot()+
  scale_x_continuous(breaks = c(1:30)) # mtry = 2 has the highest accuracy 
rfPredict<- predict(rfFit, test_cancer)
cmatRf<- confusionMatrix(rfPredict, test_cancer$diagnosis, positive = "Malignat")
cmatRf ## Accuracy of 95%

##  predicting using boosting 
gbmGrid<- expand.grid(.interaction.depth = (1:5) * 2,.n.trees = (1:10)*25, .shrinkage = c(0.01,0.05,0.1,0.5),
                      .n.minobsinnode=10,)
set.seed(100)
gbmFit<- train(diagnosis~., data = train_cancer, method="gbm", metric="Accuracy", 
               trControl=fitCtrl, tuneGrid=gbmGrid, verbose=FALSE, distribution="bernoulli",tuneLength=20)
gbmFit %>% plot()
gbmFit
gbmFit$bestTune # tuning parameters with highest accuracy on train data
gbmPredict<- predict(gbmFit,test_cancer)
cmatGbm<- confusionMatrix(gbmPredict, test_cancer$diagnosis, positive = "Malignat")
cmatGbm ##  Accuracy of 96%

##  Building Support Vector Machine
set.seed(100)
svmFit<- train(diagnosis~., data = train_cancer, method="svmRadial", metric="Accuracy", trControl=fitCtrl,tuneLength=20,
               preProcess=c("center","scale"))
svmFit
plot(svmFit, metric = "Accuracy",scales = list(x = list(log =2)))
svmPredict<- predict(svmFit, test_cancer)
cmatSvm<- confusionMatrix(svmPredict, test_cancer$diagnosis)
cmatSvm # Accuracy of 97%

##  comparing the results of the models 
model_list<- resamples(list(KNN=knnFit, LogisticReg=logFit, Rpart=rpartFit, LDA=ldaFit, QDA=qdaFit,RandomForest=rfFit,
                            GBM=gbmFit, SVM=svmFit))
summary(model_list)
##  visualize the resamples 
dotplot(model_list, metric = "Accuracy") ## GBM, SVM and KNN has highest Accuracy 

##  check to see if there is any difference in the models 
summary(diff(resamples(list(GBM=gbmFit,svm=svmFit)))) ## large pvalue indicates that there is no significant difference
summary(diff(resamples(list(GBM=gbmFit,KNN=knnFit)))) ## not significant different
summary(diff(resamples(list(SVM=svmFit,KNN=knnFit))))## SVM and KNN are not significantly different 
summary(diff(resamples(list(SVM=gbmFit,QDA=qdaFit))))

##  create a function that compares the performance of different models 
## function that to round values in a list if it is numeric 
round_num<- function(list){
  lapply(list, function(x){
    if(is.numeric(x)){
      x=round(x, 2) # round to 2 D.P
    }
    
  })
}

##  create a function that compares results of the models 

comp_summ<- function(cm, fit){
  summ<- list(TN= cm$table[1,1], #  True Negative
              TP= cm$table[2,2], #  True Positive
              FN= cm$table[1,2], #  False Negative
              FP= cm$table[2,1], #  False Positive
              Acc=cm$overall["Accuracy"], # Accuracy
              Sens=cm$byClass["Sensitivity"], # Sensitivity
              Spec=cm$byClass["Specificity"], # Specificity
              Prec=cm$byClass["Precision"], # Precision
              Recall= cm$byClass["Recall"], # Recall
              F1_Score=cm$byClass["F1"], #  F1 score
              PPV= cm$byClass["Pos Pred Value"], #  Positive predictive value
              NPV= cm$byClass["Neg Pred Value"] # Negative predictive value
  )
  round_num(summ) # rounds to 2 D.P
}

##  create a dataframe that stores the performance of the models

model_performance<- data.frame(rbind(comp_summ(cmatLog,logFit),
                                         comp_summ(cmatLda,ldaFit),
                                         comp_summ(cmatQda,qdaFit),
                                         comp_summ(cmatKnn,knnFit),
                                         comp_summ(cmatRpart,rpartFit),
                                         comp_summ(cmatSvm,svmFit),
                                         comp_summ(cmatRf,rfFit),
                                         comp_summ(cmatGbm,gbmFit)))
##  create names for rows in model performanc
rownames(model_performance)<- c("LogisticReg","LDA","QDA","KNN","RpartTree","SVM","RandomForest", "GBM")
model_performance##  SVM has the highest Accuracy, Sensitivity, Positive prred value and F1 Score.

##  a sensitivity of .98 means 98% of Malignant cancer were correctly classified. Given the benefit of correctly detecting
##  malignant cancer, this model is doing good.



