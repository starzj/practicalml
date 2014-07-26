---
title: "Practical ML - Project"
author: starzj
---

## Executive Summary

This project investigates the ability to classify human activity (sitting, standing, etc) based on wearable accelerometers. The report shows an approach to using caret to model the data.

## Model Description 

The following shows the r code to load, parition, train, and evaluate the data.

```
library(caret)

# read in remote data
inputData <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
goldTesting <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

# create data parition
inTrain <- createDataPartition(y=inputData$classe,p=0.6, list=FALSE)

# set up training/testing data sets
training <- inputData[inTrain,]
testing <- inputData[-inTrain,]

# set a seed
set.seed(7)
  
# set up training controls (5x5)
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

# run rpart training
modelFit <- train(classe ~ roll_belt+pitch_belt+yaw_belt+gyros_belt_x+gyros_belt_y+gyros_belt_z+accel_belt_x+accel_belt_y+accel_belt_z+magnet_belt_x+magnet_belt_y+magnet_belt_z+roll_arm+pitch_arm+yaw_arm+total_accel_arm+gyros_arm_x+gyros_arm_y+gyros_arm_z+accel_arm_x+accel_arm_y+accel_arm_z+magnet_arm_x+magnet_arm_y+magnet_arm_z+roll_dumbbell+pitch_dumbbell+total_accel_dumbbell+gyros_dumbbell_x+gyros_dumbbell_y+gyros_dumbbell_z+accel_dumbbell_x+accel_dumbbell_y+accel_dumbbell_z+magnet_dumbbell_x+magnet_dumbbell_y+magnet_dumbbell_z+roll_forearm+pitch_forearm+yaw_forearm+total_accel_forearm+gyros_forearm_x+gyros_forearm_y+gyros_forearm_z+accel_forearm_x+accel_forearm_y+accel_forearm_z+magnet_forearm_x+magnet_forearm_y+magnet_forearm_z, type="rpart", trControl = fitControl, data=training)
predictions <- predict(modelFit, type="raw", newdata=testing)

confusionMatrix(predictions, testing$classe)
```

## Exploratory Analysis / Variable Selection

The first challenge was to determine which variables were relevant for the classification challenge.  All variables were evaluated by using summary on the data set.  This was not particularly revelaing, but it led to further investigation of the time fields.  Intuitively, it is reasonable to believe that time of day could impact exercise method, but using qplot there was little evidence to support this.  All of the timing/windowing variables were excluded from the model.  

Given the large number of variables, it was worth investigating if more variables could be removed.  I ran a command to find columns of zero or near-zero variance.  This led to the removal of about a third of the remaining variables.  
```
nzv <- nearZeroVar(training, saveMetrics = TRUE)
```
A large number of indicators were still available for the model.  There was mild concern that there would be too many indicators, but they did provide classifying power.


## Model Selection Rationale

A number of approaches were considered including rpart, lda, rf.  Given the large number of variables and the clear variable interactions shown during exploration, recursive partitioning and regression trees was selected.  Since the results appeared sufficient, further changes to the variables were not desired.

The models were run with 5-fold, 5 time cross validation.  Ideally, these numbers would be increased, but for performance (speed) they were left small.

## Uncertainty Analysis
The results of the cross validation were quite good, as shown below.  The results were greater than 99% for accuracy, precision, recall.  There definitely is some concern about over-fitting the data.  In particular, while there is a large number of records for training they were derived from a much smaller set of individuals.  The results below represent the upper limit for performance and are likely to be lower on a different data set.

```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2227   18    0    0    0
         B    3 1491    9    0    0
         C    0    8 1356   13    2
         D    0    1    3 1272    0
         E    2    0    0    1 1440

Overall Statistics
                                          
               Accuracy : 0.9924          
                 95% CI : (0.9902, 0.9942)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9903          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9978   0.9822   0.9912   0.9891   0.9986
Specificity            0.9968   0.9981   0.9964   0.9994   0.9995
Pos Pred Value         0.9920   0.9920   0.9833   0.9969   0.9979
Neg Pred Value         0.9991   0.9957   0.9981   0.9979   0.9997
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2838   0.1900   0.1728   0.1621   0.1835
Detection Prevalence   0.2861   0.1916   0.1758   0.1626   0.1839
Balanced Accuracy      0.9973   0.9902   0.9938   0.9943   0.9991
```

## Conclusion

The result shows that this data is well suited for modeling because the variables do distinguish the various cases quite well.  The choice for variables and modeling approach here seemed suitable, but other approaches would have similar merit.

------
### References

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.
Cited by 2 (Google Scholar)

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz38aG9z2Eg

