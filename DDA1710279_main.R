############################ SVM  Digits Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

##########################################################################################################

# 1. Business Understanding: 

#This problem is about the pattern recognition of the handwritten digits(0-9) recognition. 
#The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

#########################################################################################################

# 2. Data Understanding: 
# MNIST dataset
# Number of Instances and attribites in training set: 59999 of 785 variables
# Number of Instances and attributes in testing set: 9999 of 785 variables

#3. Data Preparation: 


#Loading Neccessary libraries
# install.packages("caret")
# install.packages("kernlab")
# install.packages("dplyr")
# install.packages("readr")
# install.packages("ggplot2")
# install.packages("gridExtra")
# install.packages("caTools")

library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caTools)

#Loading Data

train_Data <- read.csv("mnist_train.csv",header = F,stringsAsFactors = T)
test_Data <- read.csv("mnist_test.csv",header = F,stringsAsFactors = T)

train<-read.csv("train.csv")
m=matrix(unlist(train_Data[10,-1]),nrow=28,byrow=TRUE)
image(m,col=grey.colors(255))

#Understanding Dimensions

dim(train_Data)
#[1] 59999   785

dim(test_Data)
#[1] 10000   785


#Structure of the dataset

str(train_Data)
str(test_Data)

#printing first few rows

head(train_Data)
head(test_Data)

#Exploring the data

summary(train_Data)
summary(test_Data)

#Checking missing value

sum(is.na(train_Data)) #There are no missing values in training data
sum(is.na(test_Data))  #There are no missing values in training data

#Checking for blanks
training_blanks <- sapply(train_Data, function(x) length(which(x == ""))) #There are no blanks in test and training datasets.
test_blanks <- sapply(test_Data, function(x) length(which(x == "")))      #There are no blanks in test and training datasets.


#Checking for outliers: As the pixel data is between 0 and 255 , anything beyond this range
# are considered outliers.Lets see the same using sapply function

train_outliers <- sapply(train_Data, function(x) length(which(x <0 & x>255)))
train_outliers #All are 0's implying no outliers.

test_outliers <-  sapply(test_Data, function(x) length(which(x <0 & x>255)))
test_outliers #All are 0's implying no outliers.


#Lets visualise a few observations

par(mfrow=c(1,2))
label_train.freq <- table(train_Data$V1)
label_test.freq <-  table(test_Data$V1)
barplot(label_train.freq)
barplot(label_test.freq)
# Shows the label classes for test and train data are well balanced

rotate <- function(x) t(apply(x, 2, rev))
m = rotate(matrix(unlist(train_Data[888,-1]),nrow = 28,byrow = T))
image(m,col=grey.colors(255))
m = rotate(matrix(unlist(train_Data[44,-1]),nrow = 28,byrow = T))
image(m,col=grey.colors(255))



# Training set  shows that there are 60K observations with 785 columns (784 if label is excluded),and the test set has 10K observations with 785 columns
# Modelling is going to be computationally intensive if the entire dataset is taken
# There are 2 approaches, either apply sampling and reduce the number of observations else perform a dimensionality reduction
# Lets apply the PCC(Principal component analysis)

# PCA also takes care of 
# a) Eliminating the redundancy in data
# b) Handling outliers
# c) Normalising the data
# d) Reduce the dimensionality of the data and handle multicollinearity

label_total <- as.factor(train_Data[[1]])  	  #converting the class column to factor type
trainreduced <- train_Data[,2:785]/255  #Normalising the data , i.e convert the pixel range from 0-255 to 0-1
traincov <- cov(trainreduced)           # Applying co-variance
trainpca <- prcomp(traincov)            # Applying the PCA via the prcomp method
plot(trainpca$sdev)					            #Visualising the SD to view the top predictors
plot(trainpca$x)						            # Visualising the x , this shows the distibution of the principal components and the distribution is elliptiical (Non-linear)
trainext <- as.matrix(trainreduced) %*% trainpca$rotation[,1:60]   #Since the PCA will be aligned in 2D, i.e convertng the wide format to long format, transposing it again for matrix multiplication
trainFinal <- data.frame(label_total,trainext)	#Getting the final dataframe on Training set

label1_total <- as.factor(test_Data$V1)
testreduced <- test_Data[,2:785]/255
testFinal <- as.matrix(testreduced) %*%  trainpca$rotation[,1:60]    # Using same dimensionality of the training data
testFinal <- data.frame(label1_total,testFinal)


#################################################################################################

# 4. Model Building

#################################################################################################

#--------------------------------------------------------------------
# 4.1 Linear model - SVM  at Cost(C) = 1
#----------------------------------------------------------------------

start.time <- Sys.time()
model_1<- ksvm(label_total ~ ., data = trainFinal,scale = FALSE,C=1)  #ksvm with C=1 as tuning parameter
#Approximate run time 5.89 minutes


# Predicting the model results 
evaluate_1<- predict(model_1, testFinal)      #predicting using the test set
confusionMatrix(evaluate_1,label1_total)		  #predicting using the test set
#Accuracy : 0.9783  (97.83%)

#--------------------------------------------------------------------
# 4.2 Linear model - SVM  at Cost(C) = 10
#-----------------------------------------------------------------------

start.time <- Sys.time()
model_2<- ksvm(label_total ~ ., data = trainFinal,scale = FALSE,C=10)     #ksvm with C=1 as tuning parameter
print(model_2)
#Approximate time.taken 4.95 minutes

# Predicting the model results 
evaluate_2<- predict(model_2, testFinal)       #predicting using the test set
confusionMatrix(evaluate_2,label1_total)		   #predicting using the test set
#Accuracy : 0.9839 (98.39%)

#--------------------------------------------------------------------
# 4.3 Using Kernel methods
#----------------------------------------------------------------------

# Using Linear Vanilla kernel


Model_linear <- ksvm(label_total~ ., data = trainFinal, scale = FALSE, kernel = "vanilladot")  #ksvm with vanilladot kernel
print(Model_linear)
#Approximate time.taken 3.36 minutes

Eval_linear   <- predict(Model_linear, testFinal)   #predicting using the test set

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,label1_total)           #predicting using the test set
#Accuracy : 0.9391  (93.91%)

#Using RBF Kernel

Model_RBF <- ksvm(label_total~ ., data = trainFinal, scale = FALSE, kernel = "rbfdot")
print(Model_RBF)
#Approximate time.taken  6.34 minutes

Eval_RBF<- predict(Model_RBF, testFinal)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,label1_total)
#Accuracy : 0.9783 (97.83%)


# ############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation.

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

#Since the cross validation on the entire dataset takes several hours of run time
#I plan to reduce not only the dimensionality of the dataset using PCA method
#but also perform a sampling using the sample_frac method of the Dplyr package, to get.

#1. Storing the sampled dataset on the dataframes on which PCA was applied, in variable called training_Sample and test_Sample
#The rationale behind having the sampling is , ideally in real world problems the testing data is given by clients, hence retained the test data as it is
#Even though an ideal rule is to have the ration of train to test as 80:20 or 70:30, in our case given the amount of observations
#Taking training sample of 33.3% or original set, hence we have around 20K observations as Train and 10K as test which is logical


set.seed(7)
train_Sample <- sample_frac(train_Data,0.3333333)
set.seed(7)
test_Sample <- sample_frac(test_Data,1)


par(mfrow=c(1,2))
label_train.freq <- table(train_Sample$V1)
label_test.freq <-  table(test_Sample$V1)
barplot(label_train.freq)
barplot(label_test.freq)

# Shows the label classes for test and train data are well balanced

#Applying the PCA procedure for the samples obtained for train and test
#The same explanation  stated above holds good for PCA

label <- as.factor(train_Sample$V1)
trainreduced <- train_Sample[,2:785]/255
traincov <- cov(trainreduced)
trainpca <- prcomp(traincov)
plot(trainpca$sdev)
trainext <- as.matrix(trainreduced) %*% trainpca$rotation[,1:60]
trainFinal_Sample <- data.frame(label,trainext)

label1 <- as.factor(test_Sample$V1)
testreduced <- test_Sample[,2:785]/255
testFinal_Sample <- as.matrix(testreduced) %*%  trainpca$rotation[,1:60]
testFinal_Sample <- data.frame(label1,testFinal_Sample)


trainControl <- trainControl(method ="cv", number = 3) #Performing a 3 fold validation


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
#Since the rbf radial kernel showed a sigma value of 0.009 , lets keep that as tuning parameter
grid <- expand.grid(.sigma=c(0.025, 0.009), .C=c(0.1,0.5,1,2) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.


fit.svm <- train(label~., data=trainFinal_Sample, method="svmRadial", metric=metric,
                 tuneGrid=grid, trControl=trainControl)

#Approximate time.taken 40 minutes

print(fit.svm)

plot(fit.svm)

######################################################################
# Checking overfitting - Non-Linear - SVM
######################################################################

# Validating the model results on test data
evaluate_non_linear<- predict(fit.svm, testFinal_Sample)
confusionMatrix(evaluate_non_linear,label1)

# Accuracy    - 0.9779

#---------------------------------MODELLING PROCESS ENDS HERE---------------------------------------------------------

######################################################################################################################
#                                SUMMARY (MODEL EXPLANATION)
#####################################################################################################################
#Problem solving methodology:
#1. Importing the test and training datasets
#2. Data preparation regarding the Missing values, outliers, blank values for both training and test sets
#3. High level visualisations to print the label data
#4. Apply dimensionality reduction technique aka PCA for both train and test set
#5. Model building using KSVM algorithm for linear models using C=1 and C=10 , and then using the Vanilla kernel and RBF kernel
#6. Note the different outputs and tuning parameter thresholds
#7. Execute the cross validation using 3 folds and with sigma as 0.025 and 0.009 using RBF kernel. This is performed on sampled dataset to acheive a better computation time
#8. Determine the final model

#Detailed summary outputs of predictions of various models on the test dataset are as below:

#Linear model, Model1: With C=1,Accuracy : 0.9783   

#                         Class 0	Class 1	Class 2	Class 3	Class 4	Class 5	Class 6	Class 7	Class 8	Class 9
# Specificity 	          0.9908	0.993	  0.9738	0.9772	0.9776	0.9776	0.9833	0.9718	0.9754	0.9613
# Sensitivity 	          0.9975	0.9985	0.997	  0.997	  0.9979	0.998	  0.9981	0.9973	0.9969	0.9977
# Balanced Accuracy     	0.9941	0.9957	0.9854	0.9871	0.9877	0.9878	0.9907	0.9846	0.9861	0.9795

#Linear model, Model2: With C=10,Accuracy : 0.9839  [Even though it looks like a highly accurate model, this also poses a risk of overfitting the data]

#                     Class: 0	Class: 1	Class: 2	Class: 3	Class: 4	Class: 5	Class: 6	Class: 7	Class: 8 	Class: 9
# Sensitivity		      0.9939		0.9938		0.9806		0.9842		0.9847		0.9832		0.9864		0.9786		0.9825		0.9703
# Specificity		      0.9981		0.9993		0.9975		0.9983		0.9982		0.9979		0.9987		0.9981		0.9978		0.9981
# Balanced Accuracy	  0.996	  	0.9966		0.9891		0.9912		0.9915		0.9905		0.9926		0.9884		0.9902		0.9842


# With linear kernel, Model_linear ,Accuracy : 0.9391  

#                        Class:0  Class:1  Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8  Class:9
# Sensitivity            0.9847   0.9885   0.9428   0.9267   0.9532   0.8823   0.9603   0.9348   0.9025   0.9039
# Specificity            0.9962   0.9963   0.9899   0.9897   0.9928   0.9912   0.9957   0.9940   0.9931   0.9935
# Balanced Accuracy      0.9905   0.9924   0.9663   0.9582   0.9730   0.9368   0.9780   0.9644   0.9478   0.9487

#With RBF kernel, Model_RBF ,Accuracy : 0.9783

#                       Class:0   Class:1  Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8  Class:9
# Sensitivity            0.9908   0.9930   0.9738   0.9772   0.9776   0.9776   0.9833   0.9718   0.9754   0.9613
# Specificity            0.9975   0.9985   0.9970   0.9970   0.9979   0.9980   0.9981   0.9973   0.9969   0.9977
# Balanced Accuracy      0.9941   0.9957   0.9854   0.9871   0.9877   0.9878   0.9907   0.9846   0.9861   0.9795

#Final model after cross validation accuracy of 0.9779, with the final values used for the model were sigma = 0.025 and C = 2.

#                        Class:0  Class:1  Class:2  Class:3  Class:4  Class:5  Class:6  Class:7  Class:8  Class:9
# Sensitivity            0.9918   0.9921   0.9758   0.9693   0.9807   0.9787   0.9791   0.9747    0.9774   0.9584
# Specificity            0.9976   0.9989   0.9953   0.9972   0.9979   0.9975   0.9982   0.9977    0.9968   0.9984
# Balanced Accuracy      0.9947   0.9955   0.9855   0.9833   0.9893   0.9881   0.9887   0.9862    0.9871   0.9784

######################################## ANALYSIS ENDS HERE ##########################################################################