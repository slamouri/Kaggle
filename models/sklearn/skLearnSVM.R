
# Kaggle Scikit-Learn Tutorial with R
# author: "Saad Lamouri"
# date: "January 31, 2015"

library(e1071)
library(caret)

# load data

# read data into memory
train <- read.csv("data/sklearn/train.csv",header = F)
trainOut <- read.csv("data/sklearn/trainLabels.csv", header = F)
test <- read.csv("data/sklearn/test.csv",header = F)

# Construct training data set
trainOut <- transform(trainOut, V1 = factor(V1))
trainData <- data.frame(train, trainOut)

predictors <- rbind(train,test)

# Explore Data Relationships
library(corrgram)
corrgram(predictors,order=NULL,lower.panel=panel.shade,
         upper.panel=NULL,text.panel = panel.txt)


# Apply Principal Component Analysis
pca <- prcomp(predictors)
summary(pca)

# Construct Training Data
components <- 14
allData <- pca$x[,1:components]


trainData <- as.data.frame(allData[1:1000,])
trainData <- data.frame(trainData, trainOut)

# Partition the Data Set
inTrain <- createDataPartition(y = trainData$V1, p = .75, list = F)

training <- trainData[inTrain,]
testing <- trainData[-inTrain,]


# Fine-tune the Model Parameters
set.seed(102)
tuned <- tune.svm(V1~., data = training, gamma = seq(.1, .5, by = .1),
                  cost = seq(1,60, by = 10))

tuned$best.parameters

# Train and Test the Model
model  <- svm(V1~., data = training, 
              gamma=tuned$best.parameters$gamma, 
              cost=tuned$best.parameters$cost, 
              type="C-classification")
summary(model)

fit <- fitted(model)
print (paste("training accuracy = ", 
             sum(fit == training[,components+1])/length(fit)))

pred <- predict(model, testing[,-(components+1)])
print (paste("testing accuracy = ", 
             sum(pred == testing[,components+1])/length(pred)))
tab <- table(pred, testing[,components+1])
tab
classAgreement(tab)


# Predict with New Data Set
finalData <- as.data.frame(allData[1001:10000,])
final <- predict(model, newdata = finalData)

fd <- data.frame(Id = 1:9000, Solution = final)

# write results
write.csv(fd, file = "data/sklearn/predictions.csv",row.names=F, quote = F)


