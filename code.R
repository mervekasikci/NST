
library("caret")
library("pROC")
library("plyr")
library("Boruta")
library("GMDH2")

set.seed(1234)

data <- read.csv("~/data.csv")

####################################################################################
#################################### Data split ####################################
####################################################################################

smp_size <- floor(0.70 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]


####################################################################################
####################### Feature Selection with BORUTA Method #######################
####################################################################################

boruta.train <- Boruta(Outcome ~ ., data = train)
result <- boruta.train$finalDecision
final <- result[result %in% c("Confirmed")]
boruta_signif <- names(boruta.train$finalDecision[boruta.train$finalDecision %in% c("Confirmed", "Tentative")])  
print(boruta_signif) 
plot(boruta.train, cex.axis=.374, las=2, xlab="", main="Variable Importance")  

train <- subset(train, select=c(boruta_signif, "Outcome"))

####################################################################################
############################## Classification Models ###############################
####################################################################################

ctrl = trainControl(method="cv", number=5, returnResamp = "final", 
                    summaryFunction = prSummary, classProbs = TRUE,
                    savePredictions = T, verbose=F)

############################ Artificial Neural Networks ############################
model_ann <- train(Outcome ~ ., data = train, method = "nnet", 
                   trControl = ctrl, metric="AUC",tuneLength = 5) 
pred_ann <- predict(model_ann, test[,-1])
perf_ann <- confMat(pred_ann, test$Outcome , verbose = FALSE)$all 
roc_ann <- roc(test$Outcome, as.numeric(pred_ann))
auc_ann <- as.numeric(roc_ann$auc)

################################### Elastic net ####################################
model_elastic = train(Outcome ~ ., data = train, method = "glmnet", 
                      trControl = ctrl, tuneLength = 5, metric="AUC")
pred_en <- predict(model_elastic, test[,-1])
perf_en <- confMat(pred_en, test$Outcome, verbose = FALSE)$all
roc_en <- roc(test$Outcome, as.numeric(pred_en))
auc_en <- as.numeric(roc_en$auc)

################################## Random Forest ###################################
model_rf <- train(Outcome ~ ., data = train,method = "rf",  
                  trControl = ctrl, metric="AUC", tuneLength = 5)
pred_rf <- predict(model_rf, test[,-1])
perf_rf <- confMat(pred_rf, test$Outcome , verbose = FALSE)$all 
roc_rf <- roc(test$Outcome, as.numeric(pred_rf))
auc_rf <- as.numeric(roc_rf$auc)

##################################### XGBoost ######################################
model_xgboost <- train(Outcome ~ ., data = train, method = "xgbTree", 
                       trControl = ctrl, metric="AUC",tuneLength = 5) 
pred_xgboost <- predict(model_xgboost, test[,-1])
perf_xgboost <- confMat(pred_xgboost, test$Outcome , verbose = FALSE)$all
roc_xgboost <- roc(test$Outcome, as.numeric(pred_xgboost))
auc_xgboost <- as.numeric(roc_xgboost$auc)

##################### Support Vector Machines - Linear Kernel ######################
model_svm_linear <- train(Outcome ~ ., data = train, method = "svmLinear", 
                          trControl = ctrl, metric="AUC",tuneLength = 5) 
pred_svm_linear <- predict(model_svm_linear, test[,-1])
perf_svm_linear <- confMat(pred_svm_linear, test$Outcome , verbose = FALSE)$all
roc_svm_linear <- roc(test$Outcome, as.numeric(pred_svm_linear))
auc_svm_linear <- as.numeric(roc_svm_linear$auc)

################## Support Vector Machines - Radial Basis Kernel ###################
model_svm_radial <- train(Outcome ~ ., data = train, method = "svmRadial", 
                          trControl = ctrl, metric="AUC",tuneLength = 5)  
pred_svm_radial <- predict(model_svm_radial, test[,-1])
perf_svm_radial <- confMat(pred_svm_radial, test$Outcome , verbose = FALSE)$all
roc_svm_radial <- roc(test$Outcome, as.numeric(pred_svm_radial))
auc_svm_radial <- as.numeric(roc_svm_radial$auc)
