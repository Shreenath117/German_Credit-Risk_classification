# Case Study Solutions : german.csv file
#---------------------------------------------------------------------------------------------------------------

library(ROCR) #Creating ROC curve
library(PRROC) #Precision-recall curve
library(glmnet) #Lasso
library(tidyverse)
library(DT)
library(glmnet)
library(rpart)
library(rpart.plot)
library(caret)
library(knitr)
library(mgcv)
library(nnet)
library(NeuralNetTools)
library(knitr)
library(dplyr)
library(tidyr)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(caret)
library(glmnet)
library(boot)
library(verification)

#---------------------------------------------------------------------------------------------------------------

# Soln. to Question 1:

# Reading the CSV file and dropping the first column
data=read.csv('german.csv')

# View the data loaded
data
# Dropping the first column which is nothing but the Serial number
data=data[2:22]
# View the dimensions (shape) of the data to be used for the analysis
dim(data)

#---------------------------------------------------------------------------------------------------------------

# Soln. to Question 2:

colnames(data) <- c("chk_acct", "duration", "credit_his", "purpose", 
                           "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                           "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                           "job", "n_people", "telephone", "foreign", "response")


#---------------------------------------------------------------------------------------------------------------
# Solution to Question 3:

#orginal response coding 1= good, 2 = bad
#we need 0 = good, 1 = bad


data$response <- data$response - 1


#---------------------------------------------------------------------------------------------------------------
# Solution to Question 4:

data$response <- as.factor(data$response)
data$response

# Levels : 0 and 1 ( 0 = good, 1 = bad)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 5:

summary(data)

str(data)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 6:

#install.packages("psych")
library(corrplot)
library(psych)

data_numeric <- data[,c('duration','amount','installment_rate','present_resid','age','n_credits')]

correlation<-corr.test(data_numeric)
correlation

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 7:

library(ggplot2)

ggplot(data, aes(factor(installment_rate), ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") + xlab("Installment Rates")

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 8:

library(dplyr)
library(reshape2)
ggplot(melt(data[,c(13,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()+ xlab("response") + ylab("age")

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 9:

ggplot(melt(data[,c(2,21)]), aes(x = variable, y = value, fill = response)) + 
  geom_boxplot()+ xlab("response") + ylab("duration")


#---------------------------------------------------------------------------------------------------------------
# Solution to Question 10:

# Observation for 7th graph:
# The installment_rate variable has a great deal of difference between the good and bad records, we see that bad records have almost the double median value than good ones.

# Observation for 8th graph :
# From the age variable, we see that the median value for bad records is lesser than that of good records, it might be premature to say young people tend to have bad credit records, but we can safely assume it tends to be riskier.

# Observation for 9th graph:
# The median value and the range of the duration variables appears to be on the higher side of bad records as compared to good records

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 11:

ggplot(data, aes(chk_acct, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 12:

ggplot(data, aes(credit_his, ..count..)) + 
  geom_bar(aes(fill = response), position = "dodge") 

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 13:

# Observation for 11th graph:
# For chk_acct we see that, the current status of the checking account matters as the frequency of the response variables is seen to differ from one sub category to another, overall A11 houses more number of bad credit records and A14 the least

# Observation for 12th graph:
# For credit_his, we observe that proportion of the response variable varies significantly, for categories A30, A31 we see the number of bad credit records are greater.

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 14:

trainrows <- sample(nrow(data), nrow(data) * 0.75)
data.train <- data[trainrows, ]
data.test <- data[-trainrows,]

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 15:

data.train.glm0 <- glm(response~., family = binomial, data.train)
summary(data.train.glm0)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 16:

#In-Sample Prediction : Response variable split
summary(data.train$response)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 17:

#Summary of In- Sample predicted values
data.train.pred<-predict(data.train.glm0,type="response")
summary(data.train.pred)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 18:

hist(predict(data.train.glm0,type="response"))

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 19:

pred <- prediction(data.train.pred, data.train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 20:

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

# Logistic Regression : 83%

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 21:

#Out of Sample Prediction
pred.glm0.test<- predict(data.train.glm0, newdata = data.test, type="response")

summary(pred.glm0.test)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 22:

library(randomForest)

m3 <- randomForest(response ~ ., data = data.train)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 23:

summary(m3)

m3_fitForest <- predict(m3, newdata = data.test, type="prob")[,2]

m3_pred <- prediction( m3_fitForest, data.test$response)
m3_perf <- performance(m3_pred, "tpr", "fpr")

#plot variable importance
varImpPlot(m3, main="Random Forest: Variable Importance")

# Model Performance plot
plot(m3_perf,colorize=TRUE, lwd=2, main = "m3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

m3_AUROC <- round(performance(m3_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_auroc
cat("AUROC: ",m3_AUROC)

# Random Forest  : 82% accuracy 