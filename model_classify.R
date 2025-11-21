library(dplyr)
library(readr)
library(caret)
library(MASS)
library(randomForest)
library(pls)

INPUT <- "data/student.csv"
student <- read_csv(INPUT, show_col_types = FALSE)

# Create pass_fail (e.g., pass if G3 >= 10)
student$pass_fail <- as.factor(ifelse(student$G3 >= 10, 1, 0))

# Train/test split
set.seed(123)
train_idx <- sample(1:nrow(student), size = 0.8 * nrow(student))
train_data <- student[train_idx, ]
test_data  <- student[-train_idx, ]

# KNN (on numeric+one-hot for categorical)
cat_cols <- sapply(train_data, is.character)
train_data[cat_cols] <- lapply(train_data[cat_cols], as.factor)
test_data[cat_cols]  <- lapply(test_data[cat_cols], as.factor)

X_train <- model.matrix(pass_fail ~ . - 1, data = train_data)
y_train <- train_data$pass_fail
X_test  <- model.matrix(pass_fail ~ . - 1, data = test_data)

knn_fit <- train(x = X_train, y = y_train, method = "knn",
                 trControl = trainControl(method = "cv", number = 5),
                 tuneGrid = expand.grid(k = 5))
knn_pred <- predict(knn_fit, newdata = X_test)
print(confusionMatrix(knn_pred, test_data$pass_fail))

# Logistic regression
logit_fit <- glm(pass_fail ~ ., data = train_data, family = binomial)
logit_prob <- predict(logit_fit, newdata = test_data, type = "response")
logit_pred <- as.factor(ifelse(logit_prob > 0.5, 1, 0))
print(confusionMatrix(logit_pred, test_data$pass_fail))

# LDA
lda_fit <- lda(pass_fail ~ ., data = train_data)
lda_pred <- predict(lda_fit, newdata = test_data)$class
print(table(lda_pred, test_data$pass_fail))

# QDA
qda_fit <- qda(pass_fail ~ ., data = train_data)
qda_pred <- predict(qda_fit, newdata = test_data)$class
print(table(qda_pred, test_data$pass_fail))

# Classification tree
library(tree)
tree_fit <- tree(pass_fail ~ ., data = train_data)
tree_prob <- predict(tree_fit, newdata = test_data)
tree_pred <- as.factor(ifelse(tree_prob[,1] > 0.5, 0, 1))
print(table(tree_pred, test_data$pass_fail))

# Bagging (RF with all predictors)
rf_bag <- randomForest(x = train_data[, setdiff(names(train_data), c("pass_fail"))],
                       y = train_data$pass_fail, ntree = 200)
bag_pred <- predict(rf_bag, newdata = test_data)
print(table(bag_pred, test_data$pass_fail))

# PCA + logistic via PCR approach (illustrative)
pcr_fit <- pcr(as.numeric(pass_fail) ~ ., data = train_data, scale = TRUE)
pcr_scores <- as.vector(predict(pcr_fit, newdata = test_data)$calibrate$calX[,1])
pcr_class <- as.factor(ifelse(pcr_scores > 0.5, 1, 0))
print(table(pcr_class, test_data$pass_fail))
