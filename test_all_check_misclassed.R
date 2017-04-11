library(LiblineaR)
library(ggplot2)
library(dplyr)
library(tidyr)
library(e1071) # for svm with prob output
library(PRROC)
library(rstackdeque)
library(purrr)
library(parallel)

# adds a "class" column of either "leaf_specific" or "root_specific" based
# on the threshold, assumes the present of column "b" (fold change) and "qval" (Q value)
# this also removes all unclassed examples if strip_unclassed is TRUE (default)
add_class <- function(input, qval_thresh, fold_thresh, strip_unclassed = TRUE) {
  input$class <- -1000
  input$class[input$qval <= qval_thresh & input$b < -1 * fold_thresh] <- 1#"leaf_specific"
  input$class[input$qval <= qval_thresh & input$b > fold_thresh] <- 0#"root_specific"
  if(strip_unclassed) {
    input <- input[!is.na(input$class), ]
  }
  return(input)
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
  print(length(train_keep_logical))
  print(dim(train_set))
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
                                type = 7,   # L2 regularized logistic regression (dual) = 7
                                cost = param,
                                bias = TRUE,  # ?? (recommended by vignette)
                                # cross = 10, # built-in cross validation; probably better to do it ourselves
                                verbose = TRUE)
  
  coefficients <- model$W
  # drop bias coefficient
  coefficients <- coefficients[1:(length(coefficients) - 1)]
  
  p <- predict(model, x_test_set_scaled, proba = TRUE, decisionValues = TRUE)
  # produce a confusion matrix
  confusion_matrix <- table(predictions = p$predictions, actuals = class_test_set)
  
  probabilities <- as.data.frame(p$probabilities)
  rownames(p$probabilities) <- rownames(test_set)
  
  coeffs_df <- data.frame(coefficients, Feature = colnames(x_train_set_scaled), stringsAsFactors = FALSE)
  auroc <- PRROC::roc.curve(probabilities[test_set$class != -1000, "1"], weights.class0 = class_test_set[test_set$class != -1000], curve = TRUE)$auc
  auprc <- PRROC::pr.curve(probabilities[test_set$class != -1000, "1"], weights.class0 = class_test_set[test_set$class != -1000], curve = TRUE)$auc.davis.goadrich
  
  retlist <- list(param = param, confusion_matrix = confusion_matrix, coeffs_df = coeffs_df, model = p, auroc = auroc, auprc = auprc)
  
  return(retlist)
}

# chosen by fair die roll, gauranteed to be random
set.seed(55) # 55 originally

setwd("~/Documents/cgrb/pis/Megraw/tss_seq_scripts/")

# load the features and differential expression data
# into all_features_diffs_wide
load("big_merged_roe_pseudoCounts_0.01_PEATcore_Hughes_NoDups_overallOC.rdat")

rownames(all_features_diffs_wide) <- all_features_diffs_wide$tss_name
# all_features_diffs_wide <- all_features_diffs_wide[,!colnames(all_features_diffs_wide) %in% c("OC_P_OVERALL_ROOT", "OC_P_OVERALL_LEAF")]
# define classes
classed_features_diffs_wide <- add_class(all_features_diffs_wide, qval_thresh = 0.05, fold_thres = 4, strip_unclassed = FALSE)

# NAs were introduced because many TSSs have overall OC features but not others
classed_features_diffs_wide <- classed_features_diffs_wide[complete.cases(classed_features_diffs_wide), ]

print("Overall class sizes:")
print(table(classed_features_diffs_wide$class))


# strip out the differential expression stuff
diffs_colnames <- c("gene_id", "pval", "qval", "b", "se_b", "mean_obs", "var_obs", 
                    "tech_var", "sigma_sq", "smooth_sigma_sq", "final_sigma_sq", 
                    "tss_name", "chr", "loc", "offset?")

# differential expression data
diff_info <- classed_features_diffs_wide[, diffs_colnames]

# features and class only
features <- classed_features_diffs_wide[, !colnames(classed_features_diffs_wide) %in% c(diffs_colnames, "class")]
classes <- classed_features_diffs_wide[, "class", drop = FALSE]


# let's break down the features into various info encoded in them...
################################################
feature_names <- data.frame(feature = colnames(features))
feature_names$type <- "other"
feature_names$type[grepl("FWD|REV", feature_names$feature)] <- "SLL"
feature_names$type[grepl("(OC_P_ROOT)|(OC_P_LEAF)", feature_names$feature)] <- "OC"

oc_features <- feature_names[feature_names$type == "OC", ]
oc_features <- extract(oc_features, feature, c("pwm", "strand", "window", "tissue"), regex = "(.+?)_(FWD|REV)_(.)_OC_P_(ROOT|LEAF)", remove = FALSE)
oc_features <- oc_features[, c("feature", "type", "pwm", "strand", "window", "tissue")]

sll_features <- feature_names[feature_names$type == "SLL", ]
sll_features <- extract(sll_features, feature, c("pwm", "strand", "window"), regex = "(.+?)_(FWD|REV)_(.)", remove = FALSE)
sll_features$tissue <- NA
sll_features <- sll_features[, c("feature", "type", "pwm", "strand", "window", "tissue")]

other_features <- feature_names[feature_names$type == "other", ]
other_features$pwm <- NA
other_features$strand <- NA
other_features$window <- NA
other_features$tissue <- NA
other_features <- other_features[, c("feature", "type", "pwm", "strand", "window", "tissue")]

feature_info <- rbind(oc_features, sll_features, other_features)
################################################

merge_by_rownames <- function(df1, df2) {
  df1$asdlkfjsdf <- rownames(df1)
  df2$asdlkfjsdf <- rownames(df2)
  outdf <- merge(df1, df2, by = "asdlkfjsdf", all = T)
  rownames(outdf) <- outdf$asdlkfjsdf
  outdf$asdlkfjsdf <- NULL
  return(outdf)
}

train_data <- cbind(features[classes$class != -1000,], class = classes[classes$class != -1000, ])
test_data <- cbind(features, classes)

result <- run_and_validate_model(0.0005, list(train_data, test_data))
colnames(result$coeffs_df) <- c("coefficient", "feature")

feature_info <- merge(feature_info, result$coeffs_df)
classes_probs <- merge_by_rownames(classes, as.data.frame(result$model$probabilities))

diffs_classes <- merge_by_rownames(diff_info, classes_probs)
colnames(diffs_classes)[colnames(diffs_classes) %in% c("b", "0", "1")] <- c("fold_change_root_over_leaf", "prob0", "prob1")

save(feature_info, features, diffs_classes, file = "featureinfo_features_diffsclasses.Rdata")


# there are three data frames that are loaded via this:
# feature_info (info about features)
# features (bigass table of feature data only)
# diffs_classes (differential expression info, assigned class, predicted class probs)
load("featureinfo_features_diffsclasses.Rdata")

# Explore the feature_info data frame, for the three types of features
print(head(feature_info[feature_info$type == "other", ]))
print(head(feature_info[feature_info$type == "SLL", ]))
print(head(feature_info[feature_info$type == "OC", ]))

# explore the differential expression results, assigned class, and probability calls for class
print(head(diffs_classes))

# check out the first 10 rows and first 10 cols of the features
print(features[1:10, 1:10])

