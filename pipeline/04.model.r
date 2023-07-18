library("pscl")
library("ROCR")
library("pROC")
library("caret")
library("glmnet")
library("dplyr") 
library("car")

#01. input file 
list <- read.table(file = '05.model_input_v2.txt', 
                   header = T,
                   sep = "\t",
                   quote = "", 
                   check.names = F,
                   stringsAsFactors = F)
head(list)[1:5,1:5]

#02. selection list
list <- list[, c("DATA","Group","41","285","292","487","545","562","564","575","582","584","622","623","729","754","839",
                 "853","1246","1352","1623","28129","28132","28901","35703","39485","52227","53407","54736",
                 "57706","61645","67824","67825","67827","69666","72361","137591","179636","208962","225992",
                 "230089","313603","318683","361575","446660","476157","544580","589436","589437","652716","693444",
                 "702967","1163710","1177574","1267768","1463165","1499973","1702221","1714373","1788301","1796616",
                 "1834196","1834207","1907578","2026787","2027290","2057025","2057741","2077137","2116657")]


#03. validation vs our
validation <- list %>%
  filter(list$DATA == "Public")

dataset <- list %>%
  filter(list$DATA == "Our")

#02. remove col
validation <- validation[,-c(1)]
dataset <- dataset[,-c(1)]

#03.identifined
table(validation$Group)
table(dataset$Group)


validation$Group <- recode(validation$Group, "'CD'=0;'UC'=1")
dataset$Group <- recode(dataset$Group, "'CD'=0;'UC'=1")

validation$Group <- as.factor(validation$Group)
table(validation$Group)
dataset$Group <- as.factor(dataset$Group)

#04.train test validation 7:3
set.seed(5380) 
training.samples <- dataset$Group %>% 
  createDataPartition(p = 0.7, 
                      list = FALSE 
  )

train.data  <- dataset[training.samples, ]
test.data <- dataset[-training.samples, ]

#05. x, y
train.data_x <- model.matrix(Group~.,train.data)[,-1]
train.data_y <- train.data$Group

test.data_x <- model.matrix(Group~.,test.data)[,-1]
test.data_y <- test.data$Group

val.data_x <- model.matrix(Group~.,validation)[,-1]
val.data_y <- validation$Group

#06.Perform 10-fold cross-validation to select lambda
ridge_cv <- cv.glmnet(x = train.data_x, 
                      y = train.data_y, 
                      alpha = 0, 
                      type.measure = "mse",
                      alignment = "lambda", 
                      family = "binomial", 
                      nfolds = 10 ) 

#07. best lambda
ridge_cv$lambda.min 
ridge_cv$lambda.1se 

#08. Plot cross-validation results
plot(ridge_cv)

#09. Best cross-validated lambda
lambda_cv <- ridge_cv$lambda.min
lambda_cv
log(ridge_cv$lambda.min) 


#08.Fit final model
model_cv <- glmnet(x = train.data_x, 
                   y = train.data_y, 
                   alpha = 0, 
                   lambda = lambda_cv, 
                   family ="binomial")

#09.predict
y_test_cv <- predict(model_cv, 
                     newx = test.data_x, 
                     type = "response", 
                     s = lambda_cv)

y_vali.data_cv <- predict(model_cv, 
                          newx = val.data_x,
                          type = "response",
                          s = lambda_cv)
y_train.data_cv <- predict(model_cv, 
                           newx = train.data_x,
                           type = "response",
                           s = lambda_cv)

#10. roc curve cutoff
roc_obj_train <- roc(train.data_y, y_train.data_cv)
coords_obj_train <- coords(roc_obj_train, "best", best.method = "youden")
auc_cutoff_train <- coords_obj_train$threshold
auc_cutoff_train

#11.cutoff value
predicted_classes_test <- ifelse(y_test_cv > auc_cutoff_train, 1, 0)  
predicted_classes_vali <- ifelse(y_vali.data_cv > auc_cutoff_train, 1, 0)  


#12. Create a confusion matrix
library(caret)
conf_matrix_test <- confusionMatrix(data = as.factor(predicted_classes_test), reference = as.factor(test.data_y))
conf_matrix_vali <- confusionMatrix(data = as.factor(predicted_classes_vali), reference = as.factor(val.data_y))

conf_matrix_test$byClass
conf_matrix_vali$byClass

#13. AUCPR
library("PRROC")
library("ROCR")
library("DMwR2")
library(precrec)

precrec_obj_test <- evalmod(scores = predicted_classes_test, labels = test.data_y)
precrec_obj_vali <- evalmod(scores = predicted_classes_vali, labels = val.data_y)
precrec_obj_test
precrec_obj_vali
