library(dplyr)
library(glmnet)
library(pls)
library(rpart)
library(randomForest)
library(gbm)
library(readr)

INPUT <- "data/student.csv"  # columns like G1, G2, G3, etc.
student <- read_csv(INPUT, show_col_types = FALSE)

# Multiple regression G3 ~ .
lm_fit <- lm(G3 ~ ., data = student)
print(summary(lm_fit))

# Ridge regression
X_all <- model.matrix(G3 ~ . , data = student)[, -1]
y <- student$G3
ridge_cv <- cv.glmnet(x = X_all, y = y, alpha = 0)
ridge_fit <- glmnet(x = X_all, y = y, alpha = 0, lambda = ridge_cv$lambda.min)
ridge_pred <- as.numeric(predict(ridge_fit, newx = X_all))
cat("Ridge MSE:", mean((ridge_pred - y)^2), "\n")

# Lasso regression (selected features)
X_sel <- as.matrix(student[, c("studytime","failures","absences","G1","G2")])
lasso_cv <- cv.glmnet(x = X_sel, y = y, alpha = 1)
lasso_fit <- glmnet(x = X_sel, y = y, alpha = 1, lambda = lasso_cv$lambda.min)
lasso_pred <- as.numeric(predict(lasso_fit, newx = X_sel))
cat("Lasso MSE:", mean((lasso_pred - y)^2), "\n")

# Partial Least Squares
pls_fit <- plsr(y ~ X_sel, ncomp = 2)
pls_pred <- as.numeric(predict(pls_fit, newdata = X_sel))
cat("PLS MSE:", mean((pls_pred - y)^2), "\n")

# Decision tree (ANOVA)
df_tree <- data.frame(X_sel, y)
tree_fit <- rpart(y ~ ., data = df_tree, method = "anova")
tree_pred <- predict(tree_fit, newdata = df_tree)
cat("Tree MSE:", mean((tree_pred - y)^2), "\n")

# Bagging
bag_fit <- randomForest(x = X_sel, y = y, ntree = 100)
bag_pred <- predict(bag_fit, newdata = data.frame(X_sel))
cat("Bagging MSE:", mean((bag_pred - y)^2), "\n")

# Random forest (regression, all numeric)
num_cols <- sapply(student, is.numeric)
rf_fit <- randomForest(x = student[, num_cols & names(student) != "G3"],
                       y = student$G3, ntree = 500)
print(rf_fit)

# Boosting
student_factorized <- student
for (c in names(student_factorized)) {
  if (is.character(student_factorized[[c]])) {
    student_factorized[[c]] <- as.factor(student_factorized[[c]])
  }
}
boost_fit <- gbm(G3 ~ ., data = student_factorized,
                 distribution = "gaussian", n.trees = 200, interaction.depth = 3)
boost_pred <- predict(boost_fit, newdata = student_factorized, n.trees = 200)
cat("Boosting MSE:", mean((boost_pred - y)^2), "\n")
