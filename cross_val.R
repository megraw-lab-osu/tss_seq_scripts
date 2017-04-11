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
#options(scipen=999)



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
  randomized_features_diffs_wide <- input[order(runif(nrow(input))), ]
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
  ret_list <- list(train_set = other_folds_bound, test_set = test_fold, validation_set = validation_fold)
  return(ret_list)
}

# given a name for a fold, and a (named) list of all the folds, extracts
# that name as the test, the rest as training (collapsed with rbind)
folds_to_train_test <- function(test_fold_name, all_folds) {
  test_fold <- all_folds[[test_fold_name]]
  other_folds <- all_folds[!names(all_folds) %in% c(test_fold_name)]
  other_folds_bound <- do.call(rbind, other_folds)
  ret_list <- list(train_set = other_folds_bound, test_set = test_fold)
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
  auroc <- PRROC::roc.curve(probabilities[,"1"], weights.class0 = class_test_set, curve = TRUE)$auc
  auprc <- PRROC::pr.curve(probabilities[,"1"], weights.class0 = class_test_set, curve = TRUE)$auc.davis.goadrich

  retlist <- list(param = param, confusion_matrix = confusion_matrix, coeffs_df = coeffs_df, model = p, auroc = auroc, auprc = auprc)
  
  return(retlist)
}


# given a fold name (length-1 character vector), uses it to extract train, validate, test sets,
# also for each param in the params list, tries that param.
# folds list is a list of data frames
# returns a list with lots of goodies
find_pstar <- function(fold_name, params_list, folds_list) {
  train_valid_test <- folds_to_train_validate_test(fold_name, folds_list)
  train_validate <- train_valid_test[c("train_set", "validation_set")]

  param_results <- lapply(params_list, run_and_validate_model, train_validate)

  best_result <- param_results[[1]]
  best_auroc <- param_results[[1]]$auroc
  for(result in param_results) {
    auroc <- result$auroc
    auprc <- result$auprc
    if(auroc > best_auroc) {
      best_result <- result
      best_auroc <- auroc
    }
  }

  all_params <- unlist(params_list)
  all_fold_names <- rep(fold_name, length(all_params))
  all_aurocs <- unlist(lapply(param_results, function(x) { return(x$auroc) } ))
  all_auprcs <- unlist(lapply(param_results, function(x) { return(x$auprc) } ))
  within_params_list <- list(fold_name = all_fold_names, param = all_params, auroc = all_aurocs, auprcs = all_auprcs)
  
  best_param <- best_result$param
  test_result <- run_and_validate_model(best_param, train_valid_test[c("train_set", "test_set")])
  ret_list <- list(fold_name = fold_name, 
                   train_set = train_valid_test$train_set,
                   test_set = train_valid_test$test_set,
                   best_model = best_result$model,
                   best_param = best_param, 
                   best_auroc = best_result$auroc, 
                   best_auprc = best_result$auprc, 
                   test_auroc = test_result$auroc, 
                   test_auprc = test_result$auprc, 
                   test_coeffs_df = test_result$coeffs_df,
                   mean_auroc = mean(all_aurocs), 
                   sd_auroc = sd(all_aurocs), 
                   within_params_list = within_params_list)
  return(ret_list)
  
}


# train_folds is a named list of data frames, possible_params is a named list of param values
# breaks out computation by fold 
# returns a list (of lists) for each fold with lots of goodies
n_fold_cross <- function(train_folds, possible_params) {
  train_folds_names <- as.list(names(train_folds))
  print(train_folds_names)
  bests_by_fold <- lapply(train_folds_names, find_pstar, possible_params, train_folds)
  return(bests_by_fold)
}



####################################
##  End functions, begin script
####################################

# chosen by fair die roll, gauranteed to be random
set.seed(55) # 55 originally

setwd("~/Documents/cgrb/pis/Megraw/tss_seq_scripts/")


# load the features and differential expression data
# into all_features_diffs_wide
load("big_merged_roe_pseudoCounts_0.01_PEATcore_Hughes_NoDups_overallOC_0_-100_with_tiled.rdat")

### : remove all but tiled features
#all_features_diffs_wide <- all_features_diffs_wide[, !(grepl("(FWD|REV)", colnames(all_features_diffs_wide)) & !grepl("tile", colnames(all_features_diffs_wide))) ]
### or, remove tiled features
all_features_diffs_wide <- all_features_diffs_wide[, !grepl("tile100", colnames(all_features_diffs_wide)) ]

# set rownames to tss names
rownames(all_features_diffs_wide) <- all_features_diffs_wide$tss_name

# debug
# all_features_diffs_wide <- all_features_diffs_wide[,!colnames(all_features_diffs_wide) %in% c("OC_P_OVERALL_ROOT", "OC_P_OVERALL_LEAF")]

# define classes, get rid of unclassed columns
classed_features_diffs_wide <- add_class(all_features_diffs_wide, qval_thresh = 0.05, fold_thres = 4)


# NAs were introduced because many TSSs have overall OC features but not others
## todo: why is this again?
classed_features_diffs_wide <- classed_features_diffs_wide[complete.cases(classed_features_diffs_wide), ]
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
folds_final_test <- split_data(classed_features_class, percent_train = 0.8, folds = 5)
train_folds <- folds_final_test$train_folds

# pstar_avg <- 0.0005

# # make it all parallel...
library(parallel)
cl <- makeCluster(6)
clusterExport(cl, list("folds_to_train_validate_test", "train_folds",
                       "run_and_validate_model"))
# replace lapply with parLapply
lapply <- function(...) {parLapply(cl, ...)}


# we'll try a bunch of different params
#possible_params <- as.list(10^seq(-6,-1,0.2))
possible_params <- as.list(seq(0.00005, 0.001, 0.00005))

print("trying params:")
print(unlist(possible_params))
names(possible_params) <- as.character(possible_params)


bests_by_fold <- n_fold_cross(train_folds, possible_params)
str(bests_by_fold[1])


################################
####### Cross val done.
################################

######### Line plots start

# make it into a table
# grab everything but the "within_params_list" entries and build a table
# map_df -> turns some parts of a list into a dataframe - from purrr library
bests_by_fold_table <- map_df(bests_by_fold, .f = function(x) {return(x[!names(x) %in% c("within_params_list", "test_coeffs_df", "train_set", "test_set", "best_model")])} )
print(bests_by_fold_table)

# grab the "within_params_list" entries and build a table
within_folds_table <- map_df(map(bests_by_fold, "within_params_list"), I)
print(as.data.frame(within_folds_table), row.names = FALSE)
#write.table(within_folds_table, file = "within_folds_param_vs_aurocs.txt", quote = F, sep = "\t", row.names = F)

df <- within_folds_table
df$fold_name <- as.character(df$fold_name)
ggplot(df) + geom_line(aes(x = param, y = auroc, color = fold_name)) +
  expand_limits(y = c(0.80, 1.0)) +
  ggtitle("ROE features and Tiled features") 


######### Line plots end


######### Run on final held out test
# geometric mean: 0.0003651741
# arithmetic mean: 0.0007218799
pstar_avg <- mean(bests_by_fold_table$best_param)
pstar_avg <- 0.0005


###########################
### Random explorations below
###########################



######## folds_final_test is a list of 2; first is a list of data frames (folds), second is the held out df
all_train <- do.call(rbind, folds_final_test$train_folds)
final_test <- folds_final_test$final_test
final_res <- run_and_validate_model(pstar_avg, list(all_train, final_test))
str(final_res)

test_calls_features <- cbind(final_res$model$probabilities, final_res$model$predictions)
colnames(test_calls_features)[1] <- "prob_class_0"
colnames(test_calls_features)[2] <- "prob_class_1"
colnames(test_calls_features)[3] <- "class_call"

all_input_test_rows <- classed_features_diffs_wide[rownames(classed_features_diffs_wide) %in% rownames(final_test), ]


#final_folds <- split_data(classed_features_class, percent_train = 1.0, folds = 8)[[1]]
#fold_names <- names(final_folds)
#fold_results <- lapply(fold_names, function(fold_name) {train_test_sets <- folds_to_train_test(fold_name, final_folds); 
#                                                        fold_result <- run_and_validate_model(pstar_avg, train_test_sets)})


# let's look at how the coeffs_df varies over folds...
# grab out each coeffs_df
coeffs_dfs_list <- map(bests_by_fold, function(x) {return(x$test_coeffs_df)})
# assign a name to each df
for(i in seq(1, length(coeffs_dfs_list))) {
  coeffs_dfs_list[[i]]$fold_num <- paste("fold", i, sep = "_")
}

# turn it into one big one
coeffs_df <- do.call(rbind, coeffs_dfs_list)
# compute the average coefficient for each feature
coeffs_df_means_by_feature <- coeffs_df %>% group_by(Feature) %>% summarize(., mean_coeff = mean(coefficients))

# get a per-feature row of coefficients for each feature
coeffs_df_wide <- spread(coeffs_df, fold_num, coefficients)
# merge in the averages
coeffs_df_wide <- merge(coeffs_df_wide, coeffs_df_means_by_feature)

# write it to a table for valerie
save(coeffs_df_wide, file = "folds_coeffs_df_wide.rdat")

# put it back into a long table for plotting/extracting
coeffs_df <- gather(coeffs_df_wide, fold_num, coefficients, -Feature, -mean_coeff)

# let's just grab the top 5% of features by average coefficient
#sub_coeffs_df <- coeffs_df[coeffs_df$mean_coeff > quantile(coeffs_df$mean_coeff, 0.99), ]
#sub_coeffs_df_wide <- coeffs_df_wide[coeffs_df_wide$mean_coeff > quantile(coeffs_df_wide$mean_coeff, 0.99), ]
# if percentile > 1, select top n
select_by_top <- function(x, percentile) {
  # there are some NA coefficients because for some folds, there isn't enough variance
  # in the columns to include them in the analysis (ed: how the F do you spell that word?)
  # we we remove them before doing the quantile shtick
  if(percentile < 1) {
    x <- x[!is.na(x$coefficients),]
    return(x[abs(x$coefficients) > quantile(abs(x$coefficients), percentile), ])
  } else {
    x <- x[!is.na(x$coefficients),]
    x <- x[rev(order(abs(x$coefficients))), ]
    return(x[seq(1,percentile),])
  }
}
sub_coeffs_df <- coeffs_df %>% group_by(., fold_num) %>% do(., select_by_top(., 20))
sub_coeffs_counts <- sub_coeffs_df %>% group_by(Feature) %>% summarize(., count_occurances = length(coefficients))
sub_coeffs_df <- merge(sub_coeffs_df, sub_coeffs_counts, all = T)
sub_coeffs_df_wide <- spread(sub_coeffs_df, fold_num, coefficients)

# for Molly, table form
sub_coeffs_df_wide <- sub_coeffs_df_wide[rev(order(sub_coeffs_df_wide$count_occurances)),]
#write.table(sub_coeffs_df_wide[,c("Feature", "count_occurances")], file = "top20_count_fold_occurances.txt",
#            quote = F, sep = "\t", row.names = F)

library(plotly)
p <- ggplot() +
  geom_point(data = sub_coeffs_df, mapping = aes(x = fold_num, y = coefficients, color = Feature)) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_1", xend = "fold_2", y = fold_1, yend = fold_2,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_1", xend = "fold_3", y = fold_1, yend = fold_3,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_1", xend = "fold_4", y = fold_1, yend = fold_4,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_1", xend = "fold_5", y = fold_1, yend = fold_5,
                                                        color = Feature), alpha = 0.1) +
  
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_2", xend = "fold_3", y = fold_2, yend = fold_3,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_2", xend = "fold_4", y = fold_2, yend = fold_4,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_2", xend = "fold_5", y = fold_2, yend = fold_5,
                                                        color = Feature), alpha = 0.1) +
  
  
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_3", xend = "fold_4", y = fold_3, yend = fold_4,
                                                        color = Feature), alpha = 0.1) +
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_3", xend = "fold_5", y = fold_3, yend = fold_5,
                                                        color = Feature), alpha = 0.1) +
  
  geom_segment(data = sub_coeffs_df_wide, mapping = aes(x = "fold_4", xend = "fold_5", y = fold_4, yend = fold_5,
                                                        color = Feature), alpha = 0.1) +
  #expand_limits(y = 0) +
  theme(legend.position = "none") +
  facet_grid(count_occurances ~ .) +
  guides(color = "none")
plot(p)

#x <- plot_ly(username = "oneilsh", key = "fb7llMt4tf0sk5eige0v")
#ggplotly(p)
#plotly_POST(p)



# let's build a data frame of probability/fold names
folds_probs_stack <- rstack()
for(fold_num in seq(1,length(bests_by_fold))) {
  probs <- bests_by_fold[[fold_num]]$best_model$probabilities
  probs <- as.data.frame(probs)
  colnames(probs) <- c("prob_0", "prob_1")
  probs$fold_num <- fold_num
  folds_probs_stack <- insert_top(folds_probs_stack, probs)
}
folds_probs_df <- do.call(rbind, as.list(folds_probs_stack))

print(head(folds_probs_df))


folds_probs_df$tss_name <- rownames(folds_probs_df)
classed_diffs_info_calls <- merge(classed_diffs_info, folds_probs_df)
#write.table(classed_diffs_info_calls, file = "classed_diffs_info_calls.txt", quote = F, sep = "\t", row.names = F)



# build a dataframe row-by-row example

library(rstackdeque)
mystack <- rstack()
for(i in seq(1,1000)) {
  df_row <- data.frame(val = i, logval = log(i))
  mystack <- insert_top(mystack, df_row)
}
print(mystack)
mystack_df_conv <- as.data.frame(mystack)




