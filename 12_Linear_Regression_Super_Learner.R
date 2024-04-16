install.packages("VIM")
install.packages("SuperLearner")
install.packages("caTools")
install.packages("arm")
install.packages("glmnet")
install.packages("randomForest")
library(tidyverse)
library(VIM) 
library(ggplot2)
library(SuperLearner)
library(caTools)
library(arm)
library(glmnet)
library(randomForest)

data <- read.csv("/Users/scottye/Desktop/Datathon/complete_df_filtered.csv")

selected_vars <- c('los',
                  'anchor_age',
                  'icu_stay_num',
                  'admission_num',
                   'gender',
                   'procedure_freq',
                   'in2first_days',
                   'death',
                   'initial_vasopressor_needed_within_24hr',
                   'initial_ventilation_within_6hr',
                   'initial_fio2_above_60_within_6hr',
                   'initial_sodium',
                   'initial_wbc',
                   'initial_creatinine',
                   'initial_ph',
                   'initial_lactate',
                   'initial_baseexcess',
                   'initial_glucose',
                   'initial_crp',
                   'last_fio2_above_60',
                   'last_ph',
                   'last_lactate',
                   'last_baseexcess',
                   'last_hemoglobin',
                   'last_creatinine',
                   'last_sodium',
                   'last_glucose',
                   'last_wbc',
                   'last_crp',
                   'readiness')

# Subset the data and replace NAs with mean
sub_data <- data[selected_vars]
for (var in selected_vars) {
  if (is.numeric(sub_data[[var]])) {  
    mean_value <- mean(sub_data[[var]], na.rm = TRUE)  
    sub_data[[var]][is.na(sub_data[[var]])] <- mean_value  
  }
}

# Fit the linear model
model <- lm(los ~ ., data = sub_data)
summary(model)

vif_values <- vif(model)
print(vif_values)

# Check Linearity Assumption
ggplot(data = model.frame(model), aes(x = fitted(model), y = residuals(model))) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(title = "Residuals vs Fitted", x = "Fitted Values", y = "Residuals") +
  theme_minimal()

# Homogeneity of Variance
ggplot(data = model.frame(model), aes(x = fitted(model), y = sqrt(abs(residuals(model))))) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(title = "Scale-Location Plot", x = "Fitted Values", y = "Square Root of Absolute Residuals") +
  theme_minimal()

# Normal
ggplot(data.frame(StandardizedResiduals = rstandard(model)), aes(sample = StandardizedResiduals)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Normal Q-Q Plot", x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal()

# residuals vs. leverage
ggplot(data = model.frame(model), aes(x = hatvalues(model), y = rstandard(model))) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(title = "Residuals vs Leverage", x = "Leverage", y = "Standardized Residuals") +
  theme_minimal() +
  geom_vline(xintercept = 2 * mean(hatvalues(model)), linetype = "dotted", color = "blue")


# Ensemble Super Learner
data <- read.csv("/Users/scottye/Desktop/Datathon/complete_df_filtered.csv")

selected_vars <- c('los', 'procedure_freq', 'in2first_days', 
                   'initial_vasopressor_needed_within_24hr', 'initial_sodium',
                   'initial_fio2_above_60_within_6hr', 'initial_wbc',
                   'initial_creatinine', 'last_baseexcess', 
                   'last_crp')

sub_data <- data[selected_vars]
for (var in selected_vars) {
  if (is.numeric(sub_data[[var]])) {  
    mean_value <- mean(sub_data[[var]], na.rm = TRUE)  
    sub_data[[var]][is.na(sub_data[[var]])] <- mean_value  
  }
}
split <- sample.split(sub_data$los, SplitRatio = 0.7)  # 70% for training

train <- sub_data[split, ]
test <- sub_data[!split, ]

trainX <- train[, !(names(train) %in% "los")]
trainY <- train$los
testX <- test[, !(names(test) %in% "los")]
testY <- test$los

# Discrete lm super learner

# sl_lm = SuperLearner(Y = trainY, X = trainX, family = gaussian(), SL.library = "SL.lm")
# best_fold <- which.min(sl_lm$cvRisk)
# lm_model <- sl_lm$fitLibrary$SL.lm_All$object[[best_fold]]

# LOS = (Intercept) + (coeff1 * x1) + (coeff2 * x2) + ... + (coeffN * xN)

# Ensemble super learner
sl = SuperLearner(Y = trainY, X = trainX, 
                  family = gaussian(), SL.library = c("SL.randomForest", "SL.glmnet",
                                                      "SL.bayesglm","SL.svm")) 

predictions <- predict(sl, newdata = testX)$pred

# Testing
test_R2 <- cor(testY, predictions)^2
mse <- mean((testY - predictions)^2)
rmse <- sqrt(mse)
mae <- mean(abs(testY - predictions))

# Print additional performance metrics
print(paste("Testing R-squared: ", round(test_R2, 4)))
print(paste("Testing MSE: ", round(mse, 4)))
print(paste("Testing RMSE: ", round(rmse, 4)))
print(paste("Testing MAE: ", round(mae, 4)))

model <- lm(los ~ ., data = sub_data)
summary(model)

