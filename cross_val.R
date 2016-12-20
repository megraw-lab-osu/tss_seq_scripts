library(LiblineaR)
library(ggplot2)
library(dplyr)
library(tidyr)
library(e1071) # for svm with prob output
library(PRROC)
library(rstackdeque)
library(purrr)
library(parallel)


rm(list = ls())
options(scipen=999)





geometric_mean <- function(x) exp(mean(log(x)))


# adds a "class" column of either "leaf_specific" or "root_specific" based
# on the threshold, assumes the present of column "b" (fold change) and "qval" (Q value)
# this also removes all unclassed examples
add_class <- function(input, qval_thresh, fold_thresh) {
  input$class <- NA
  input$class[input$qval <= qval_thresh & input$b < -1 * fold_thresh] <- 1#"leaf_specific"
  input$class[input$qval <= qval_thresh & input$b > fold_thresh] <- 0#"root_specific"
  input <- input[!is.na(input$class), ]
  return(input)
}



# split dataframe into train (and this into folds), final test
# works on any data frame
# rows will be randomized first
# returns a list; first el being a list of folds (data frames), second being the final test data frame
split_data <- function(input, percent_train = 0.8, folds = 5) {
  randomized_features_diffs_wide <- input[order(runif(nrow(classed_features_diffs_wide))), ]
  train_index <- as.integer(percent_train * nrow(randomized_features_diffs_wide))
  train_features_diffs_wide <- randomized_features_diffs_wide[seq(1, train_index), ]
  final_test_features_diffs_wide <- randomized_features_diffs_wide[seq(train_index + 1, nrow(randomized_features_diffs_wide)), ]
  
  folds_list <- split(train_features_diffs_wide, seq(1,nrow(train_features_diffs_wide)) %% folds)
  ret_list <- list(train_folds = folds_list, final_test = final_test_features_diffs_wide)
  return(ret_list)
}



# given a name for a fold, and a (named) list of all the folds, extracts
# that name as the test, the next name as the validation,
# and the rest as training (collapsed with rbind)
folds_to_train_validate_test <- function(test_fold_name, all_folds) {
  validation_fold_index <- (which(names(train_folds) == test_fold_name) + 1) %% length(all_folds) + 1
  validation_fold_name <- names(all_folds)[validation_fold_index]
  test_fold <- all_folds[[test_fold_name]]
  validation_fold <- all_folds[[validation_fold_name]]
  other_folds <- all_folds[!names(all_folds) %in% c(test_fold_name, validation_fold_name)]
  other_folds_bound <- do.call(rbind, other_folds)
  ret_list <- list(test_set = test_fold, validation_set = validation_fold, train_set = other_folds_bound)
  return(ret_list)
}

# runs the model on the given train_validate list (2 data frames), using regularization parameter param.
# note that this ASSUMES that train_validate_test_list is a list of 2 data frames,
# and each of those data frames has only numeric columns and a "class" column (factor) to be
# predicted
# returns: list of 6: param, confusion_matrix, coeffs_df, model, auroc, auprc
run_and_validate_model <- function(param, train_validate) {
  train_set <- train_validate[[1]]
  test_set <- train_validate[[2]]
  # hm, we gotta get rid of the cols that are all identical if there are any (0s sometimes)
  # but will this F it up? Might make comparisons after the fact tricky...
  # also we gotta not do this determination based on the "class" column
  train_keep_logical <- !unlist(lapply(train_set[,colnames(train_set) != "class"], function(col){sd(col) == 0}))
  train_keep_logical <- c(train_keep_logical, TRUE)
  train_set <- train_set[, train_keep_logical]
  test_set <- test_set[, train_keep_logical]
  
  x_train_set <- train_set[, !colnames(train_set) %in% "class"]
  class_train_set <- train_set$class
  
  x_test_set <- test_set[, !colnames(test_set) %in% "class"]
  class_test_set <- test_set$class
  
  # scale the data
  x_train_set_scaled <- scale(x_train_set, center = TRUE, scale = TRUE)
  # also scale the test set by the same scale factor
  x_test_set_scaled <- scale(x_test_set, attr(x_train_set_scaled, "scaled:center"), attr(x_train_set_scaled, "scaled:scale"))
  
  model <- LiblineaR::LiblineaR(data = x_train_set_scaled, 
                     target = class_train_set,
                     type = 7,   # L2 regularized logistic regression (dual)
                     cost = param,
                     bias = TRUE,  # ?? (recommended by vignette)
                     # cross = 10, # built-in cross validation; probably better to do it ourselves
                     verbose = FALSE)
  
  coefficients <- model$W
  # drop bias coefficient
  coefficients <- coefficients[1:(length(coefficients) - 1)]
  
  p <- predict(model, x_test_set_scaled, proba = TRUE, decisionValues = TRUE)
  # produce a confusion matrix
  confusion_matrix <- table(predictions = p$predictions, actuals = class_test_set)
  
  probabilities <- p$probabilities

  coeffs_df <- data.frame(coefficients, Feature = colnames(x_train_set_scaled), stringsAsFactors = FALSE)
  auroc <- PRROC::roc.curve(probabilities[,"1"], weights.class0 = class_test_set, curve = TRUE)$auc
  auprc <- PRROC::pr.curve(probabilities[,"1"], weights.class0 = class_test_set, curve = TRUE)$auc.davis.goadrich

  retlist <- list(param = param, confusion_matrix = confusion_matrix, coeffs_df = coeffs_df, model = p, auroc = auroc, auprc = auprc)
  
  return(retlist)
}


# given a fold name, uses it to extract train, validate, test sets,
# also for each param in the params list, tries that param.
find_pstar <- function(fold_name, params_list, folds_list) {
  train_valid_test <- folds_to_train_validate_test(fold_name, folds_list)
  train_validate <- train_valid_test[c("train_set", "validation_set")]

  param_results <- lapply(params_list, run_and_validate_model, train_validate)
  best_result <- param_results[[1]]
  best_auroc <- param_results[[1]]$auroc
  for(result in param_results) {
    auroc <- result$auroc
    if(auroc > best_auroc) {
      best_result <- result
      best_auroc <- auroc
    }
  }
  
  best_param <- best_result$param
  test_result <- run_and_validate_model(best_param, train_valid_test[c("train_set", "test_set")])
  ret_list <- list(fold_name = fold_name, best_param = best_param, best_auroc = best_result$auroc, best_auprc = best_result$auprc, test_auroc = test_result$auroc, test_auprc = test_result$auprc)
  return(ret_list)
  
}

n_fold_cross <- function(train_folds, possible_params) {
  train_folds_names <- as.list(names(train_folds))
  bests_by_fold <- lapply(train_folds_names, find_pstar, possible_params, train_folds)
  return(bests_by_fold)
}



####################################
##  End functions, begin script
####################################

# chosen by fair die roll, gauranteed to be random
set.seed(55)


# load the features and differential expression data
# into all_features_diffs_wide
load("big_merged_roe_pseudoCounts_0.01_PEATcore_Hughes_NoDups.rdat")

# define classes
classed_features_diffs_wide <- add_class(all_features_diffs_wide, qval_thresh = 0.05, fold_thres = 4)
print("Overall class sizes:")
print(table(classed_features_diffs_wide$class))


# strip out the differential expression stuff
diffs_colnames <- c("gene_id", "pval", "qval", "b", "se_b", "mean_obs", "var_obs", 
                    "tech_var", "sigma_sq", "smooth_sigma_sq", "final_sigma_sq", 
                    "tss_name", "chr", "loc", "offset?")
# differential expression data
classed_diffs_info <- classed_features_diffs_wide[, diffs_colnames]
# features and class only
classed_features_class <- classed_features_diffs_wide[, !colnames(classed_features_diffs_wide) %in% diffs_colnames]



# split into 80% 8-fold set, and 20% final test
folds_final_test <- split_data(classed_features_class, percent_train = 0.8, folds = 8)
train_folds <- folds_final_test$train_folds


# make it all parallel...
library(parallel)
cl <- makeCluster(6)
clusterExport(cl, list("folds_to_train_validate_test", "train_folds",
                       "run_and_validate_model"))
lapply <- function(...) {parLapply(cl, ...)}


# we'll try a bunch of different params 
possible_params <- as.list(10^seq(-6,-1,0.2))
print("trying params:")
print(unlist(possible_params))
names(possible_params) <- as.character(possible_params)


bests_by_fold <- n_fold_cross(train_folds, possible_params)

# make it into a table
bests_by_fold_table <- map_df(bests_by_fold, I)
print(bests_by_fold_table)
# geometric mean: 0.0003651741
# arithmetic mean: 0.0007218799
pstar_avg <- mean(bests_by_fold_table$best_param)


all_train <- do.call(rbind, folds_final_test$train_folds)
final_test <- folds_final_test$final_test
final_res <- run_and_validate_model(pstar_avg, list(all_train, final_test))
str(final_res)
