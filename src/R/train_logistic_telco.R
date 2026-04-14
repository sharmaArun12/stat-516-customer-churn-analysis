# 1. Setup & Better Splitting Strategy
library(tidyverse)
library(caret)   # Added for stratified splitting and metrics
library(glmnet)  # Added for Penalized Regression
library(pROC)    # Added for AUC evaluation

# Load data
file_path <- "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
churn_data <- read.csv(file_path, na.strings = c(" ", "", "NA"), stringsAsFactors = FALSE)

# Clean Data (Keeping the continuous variables this time)
churn_data_clean <- churn_data %>%
  na.omit() %>%
  select(-customerID) %>% # Only drop the ID
  mutate(
    MultipleLines = ifelse(MultipleLines == "No phone service", "No", MultipleLines),
    OnlineSecurity = ifelse(OnlineSecurity == "No internet service", "No", OnlineSecurity),
    OnlineBackup = ifelse(OnlineBackup == "No internet service", "No", OnlineBackup),
    DeviceProtection = ifelse(DeviceProtection == "No internet service", "No", DeviceProtection),
    TechSupport = ifelse(TechSupport == "No internet service", "No", TechSupport),
    StreamingTV = ifelse(StreamingTV == "No internet service", "No", StreamingTV),
    StreamingMovies = ifelse(StreamingMovies == "No internet service", "No", StreamingMovies),
    
    # Convert character columns to factors en masse
    across(where(is.character), as.factor),
    
    # Keep Churn as a factor for classification packages, but 1/0 is fine for glmnet
    Churn_Binary = ifelse(Churn == "Yes", 1, 0)
  ) %>% select(-Churn)

# 2. Stratified Splitting
set.seed(123)
# createDataPartition ensures the 70/30 split maintains the exact ratio of Churners
train_index <- createDataPartition(churn_data_clean$Churn_Binary, p = 0.7, list = FALSE)
train_data <- churn_data_clean[train_index, ]
test_data  <- churn_data_clean[-train_index, ]

# Prepare Matrices for glmnet
# model.matrix creates dummy variables and removes the intercept column (-1)
x_train <- model.matrix(Churn_Binary ~ ., data = train_data)[, -1]
y_train <- train_data$Churn_Binary

x_test <- model.matrix(Churn_Binary ~ ., data = test_data)[, -1]
y_test <- test_data$Churn_Binary

# ---------------------------------------------------------
# Paradigm 1: Ridge Regression (Alpha = 0)
# Shrinks coefficients to handle collinearity, keeps all variables
# ---------------------------------------------------------
# cv.glmnet automatically performs k-fold cross-validation to find the optimal penalty (lambda)
ridge_cv <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial", type.measure = "auc")

# Predict on test set using the best lambda
ridge_pred <- predict(ridge_cv, s = "lambda.min", newx = x_test, type = "response")

# ---------------------------------------------------------
# Paradigm 2: Lasso Regression (Alpha = 1)
# Performs continuous variable selection by shrinking some coefficients to exactly 0
# ---------------------------------------------------------
lasso_cv <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", type.measure = "auc")
lasso_pred <- predict(lasso_cv, s = "lambda.min", newx = x_test, type = "response")

# See which variables Lasso kept (non-zero coefficients)
lasso_coefs <- coef(lasso_cv, s = "lambda.min")
print("--- Lasso Selected Variables ---")
print(lasso_coefs[lasso_coefs[, 1] != 0, ])

# ---------------------------------------------------------
# Paradigm 3: Elastic Net (Alpha between 0 and 1)
# ---------------------------------------------------------
# Often, you tune alpha via a grid search, but let's test a balanced 0.5
enet_cv <- cv.glmnet(x_train, y_train, alpha = 0.5, family = "binomial", type.measure = "auc")
enet_pred <- predict(enet_cv, s = "lambda.min", newx = x_test, type = "response")

# ---------------------------------------------------------
# The 3-Tier Evaluation Engine
# ---------------------------------------------------------
# We create a function to ensure rigorous, repeatable benchmarking
evaluate_model <- function(actuals, predictions, threshold = 0.5, model_name = "Model") {
  
  # Crucial: glmnet outputs a matrix. We MUST force it to a numeric vector for PRROC and DescTools
  probs <- as.numeric(as.vector(predictions))
  
  # --- Tier 1: Rank-Ordering ---
  roc_curve <- roc(actuals, probs, quiet = TRUE)
  roc_auc <- auc(roc_curve)
  
  pr_curve <- pr.curve(scores.class0 = probs[actuals == 1], 
                       scores.class1 = probs[actuals == 0], 
                       curve = FALSE)
  pr_auc <- pr_curve$auc.integral
  
  # --- Tier 2: Calibration ---
  brier_score <- BrierScore(actuals, probs)
  
  # --- Tier 3: Hard Classification ---
  preds_binary <- ifelse(probs > threshold, 1, 0)
  
  # Use caret for confusion matrix metrics
  conf_mat <- confusionMatrix(factor(preds_binary, levels = c(0,1)), 
                              factor(actuals, levels = c(0,1)), 
                              positive = "1")
  f1_score <- conf_mat$byClass['F1']
  
  # Return a clean data frame
  return(data.frame(
    Model = model_name,
    ROC_AUC = round(roc_auc, 4),
    PR_AUC = round(pr_auc, 4),
    Brier = round(brier_score, 4),
    F1 = round(f1_score, 4)
  ))
}

# ---------------------------------------------------------
# Execute Benchmarks
# ---------------------------------------------------------
# Notice how clean the execution is now. 
# We are currently assuming a 0.5 threshold for Tier 3, which is highly debatable for churn.
results_ridge <- evaluate_model(y_test, ridge_pred, threshold = 0.3, model_name = "Ridge")
results_lasso <- evaluate_model(y_test, lasso_pred, threshold = 0.3, model_name = "Lasso")
results_enet  <- evaluate_model(y_test, enet_pred, threshold = 0.3, model_name = "Elastic Net")

# Bind them together into a final benchmarking table
final_benchmark <- bind_rows(results_ridge, results_lasso, results_enet)
print(final_benchmark)
