#Install and load the required libraries 
install.packages("data.table")
install.packages("xgboost")
install.packages("caret")
install.packages("dpylr")
install.packages("ggplot2")

#Load required libraries
library(data.table)
library(xgboost)
library(caret)
library(dpylr)
library(ggplot2)

#Load the Pima-Indians-Diabetes data set
url <- "../EsembleML/Boosting/data/diabetes.csv"
colnames <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome")
diabetes_data <- fread(url, skip = 16, header = FALSE, col.names = colnames)

#Display the first few rows of the data set
head(diabetes_data)

#Explore the data set
#Check for missing data
missing_data <- colSums(is.na(diabetes_data))
missing_data

#Create a correlation heatmap for the data
correlation_matrix <- abs(cor(diabetes_data))

#Now plot the data
plt <- ggplot(data = as.data.frame(as.table(correlation_matrix)), aes(x = Var1, y = Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradientn(colors = colorRampPalette(c("purple", "blue", "red"))(50), name = "Correlation") +
  geom_text(aes(label = round(Freq, 2)), color = "white", size=4) +
  theme_minimal() +
  theme(axis.text.x = element_text(size=10, angle = 45, hjust = 1), 
        axis.text.y = element_text(size = 10),
        plot.title = element_text(size = 10),   # Adjust plot title font size
        legend.text = element_text(size = 10),
        legend.title = element_text(size=10))
print(plt)

#Show the class-wise distribution of the data set
class_counts <- table(diabetes_data$Outcome)
barplot(class_counts,
        names.arg = c("Tested Negative", "Tested Positive"),
        col = c("blue", "red"),
        xlab = "Class", ylab = "Number of Instances",
        main = "Class wise Distribution of Diabetes Dataset")

#Extract required features
X_data <- diabetes_data[, c("Pregnancies", "Glucose", "BMI", "Age", "Insulin", "DiabetesPedigreeFunction"), with = FALSE]
y_data <- diabetes_data$Outcome

#Split the data set
set.seed(50)

# 80/20 split
split_indices <- createDataPartition(y_data, p = 0.8, list = FALSE)
X_train <- X_data[split_indices, ]
X_test <- X_data[-split_indices, ]
y_train <- y_data[split_indices]
y_test <- y_data[-split_indices]

# Train and evaluate the model with default hyperparameters
default_model <- xgboost(data = as.matrix(X_train),
                         label = y_train,
                         booster = "gbtree",
                         objective = "binary:logistic",
                         nrounds = 100,
                         verbose = 0)

y_pred <- predict(default_model, as.matrix(X_test), type = "response") > 0.5
accuracy <- sum(y_pred == y_test) / length(y_test)

#Use a grid search of hyperparameters and perform cross validation
hyperparam_grid <- expand.grid(
  nrounds = seq(from = 100, to = 300, by = 100),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(4, 5, 6),
  gamma = c(0, 1, 2),
  colsample_bytree = c(0.5, 0.75, 1.0),
  min_child_weight = c(1, 3, 5),
  subsample = 1
)

#we’ll set up the configuration for the cross validation
tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 4, # with n folds
  verboseIter = FALSE, # no training log
  allowParallel = FALSE
)

bst <- caret::train(
  x = X_train,
  y = as.factor(y_train),
  trControl = tune_control,
  tuneGrid = hyperparam_grid,
  method = "xgbTree", #  this says we want XGB
  verbose = FALSE,
  verbosity = 0
)

#We can see the best hyperparameters
bst$bestTune$

#We’ve found the best combination of hyperparameters
final_model <- xgboost(data = as.matrix(X_train),
                       label = y_train,
                       booster = "gbtree",
                       objective = "binary:logistic",
                       nrounds = bst$bestTune$nrounds,
                       max_depth = bst$bestTune$max_depth,
                       colsample_bytree = bst$bestTune$colsample_bytree,
                       min_child_weight = bst$bestTune$min_child_weight,
                       subsample = bst$bestTune$subsample,
                       eta = bst$bestTune$eta,
                       gamma = bst$bestTune$gamma,
                       scale_pos_weight = 0.5, # because our dataset is unbalanced
                       verbose = 0)

#now evaluate
y_pred <- predict(final_model, as.matrix(X_test), type = "response") > 0.5
accuracy <- sum(y_pred == y_test) / length(y_test)

#We can easily see how much different features affect the output of the model overall
importance_matrix <- xgb.importance(colnames(X_train), model = final_model)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")

#Ref: https://developer.ibm.com/tutorials/awb-implement-xgboost-in-r/
