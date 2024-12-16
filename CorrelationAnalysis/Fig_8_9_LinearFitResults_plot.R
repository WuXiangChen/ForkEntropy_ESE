library("forecast")
library("dplyr")
library("tidyverse")
library("knitr")
library("stringr")
library("lme4")
library("lmerTest")
library("performance")
library("car")
library("pscl")
library("ggplot2")
library("MASS")
library("mlmRev")
library("agridat")
library("MCMCglmm")
library("plotMCMC")
library("see")
library("patchwork")
library("tseries")
library("car")
library("timeSeries")
library('urca')
library("Matrix")
library("astsa")
library("lme4")
library("writexl")



source("utils/fitted_methods.R")
source("utils/generate_and_save_data.R")
source("utils/generate_data.R")
source("utils/validation_utils.R")

Sys.setlocale(category = 'LC_ALL', locale = 'English_United States.1252')

raw_data <- read.csv(
  "fork_entropy_dataset_extended2.csv",
  sep = ",",
  header = TRUE,
  encoding = "UTF-8",
  stringsAsFactors = FALSE)

#定义一个函数，接收data 参数
get_sample_count_of_each_project <- function(data) {
  projects <- unique(data$project) # 取列进行唯一化展示
  print(paste(
    "The dataset includes ", length(projects),
    " projects. The sample counts of projects are as below:", sep = ""))
  for (p in projects) {
    n <- nrow(subset(data, project == p)) # 计算每个项目的样本数量
    print(paste(p, ": ", n, sep = ""))
  }
}

# 逆差分
p2p_inverse_difference <- function(diff_series, init_value, differences = 1) {
  # diff_series: 差分后的序列
  # original_series: 原始序列（用于提供初始值）
  # differences: 差分的阶数
  n1 <- length(diff_series)
  n2 <- length(init_value)
  # 如果n1和n2是等长的，那么应该从第二个差分值进行计算
  if (n1==n2){
    xi <- init_value
    # 逆差分操作
    result <- c(xi[1])
    # 使用索引循环
    for (i in 2:length(diff_series)) {
      temp <- diff_series[i] + xi[i-1]
      result <- c(result, temp)
    }
  }else if(n2==n1+1){
    xi <- init_value
    # 逆差分操作
    result <- c(xi[1])
    # 使用索引循环
    for (i in 1:length(diff_series)) {
      temp <- diff_series[i] + xi[i]
      result <- c(result, temp)
    }
  }else{
    print("p2pInverseError")
  }
  # check length
  if (length(result)!=n2){
    print("p2pInverseError")
  }
  return(result)
}

recover_original_timeseries <- function(original_data_dv, predicted_values, DV_name){
  dv_and_pro_df <- original_data_dv[, c(DV_name, 'pro_name')]
  unique_pro_names <- unique(dv_and_pro_df$pro_name)
  # 延迟增加的时候 init_value好像有变异性！
  handled_result <- c()
  actual_result <- c()
  # 分组还原
  for (current_pro_name in unique_pro_names) {
    selected_columns <- dv_and_pro_df[dv_and_pro_df$pro_name == current_pro_name,DV_name ]
    predict_selected_columns <- predicted_values[predicted_values$pro_name == current_pro_name,"predicted_values" ]
    n1 <- length(predict_selected_columns)
    n2 <- length(selected_columns)
    if(n1 <(n2-1)){
        next
    }
    recovered_series <- p2p_inverse_difference(diff_series = predict_selected_columns, init_value = selected_columns)

    actual_result <- c(actual_result,selected_columns)
    handled_result <- c(handled_result,recovered_series)
    # print(paste("Processing pro_name:", pro_name, "at row indices:", toString(row_indices)))
  }
  return (list(handled_result=handled_result,actual_result=actual_result))
}

calculate_residuals <- function(original_data_dv, predicted_values_all, DV_name) {
  dv_and_pro_df <- original_data_dv[, c(DV_name, 'pro_name')]
  unique_pro_names <- unique(dv_and_pro_df$pro_name)

  result <- lapply(unique_pro_names, function(name) {
    indices <- which(dv_and_pro_df$pro_name == name)
    pro_len <- length(indices)
    return(data.frame(pro_name = name, row_indices = indices[1], pro_len = pro_len))
  })

  result_df <- do.call(rbind,result)

  pre <- 1
  all_rmse <- 0
  for (i in 1:nrow(result_df)) {
    # 获取当前行的 pro_name 和 row_indices
    current_row <- result_df[i, ]
    name <- current_row$pro_name
    row_indices <- current_row$row_indices
    pro_len <- current_row$pro_len

    pre_index <- pre+pro_len-1

    actual_values <- as.numeric(subset(dv_and_pro_df, pro_name == name)[[DV_name]])
    predicted_values <- predicted_values_all[pre:pre_index]
    residuals <- actual_values - predicted_values
    rmse_part <- sqrt(mean(residuals^2,na.rm = TRUE))
    pre <- pre_index+1
    all_rmse <- rmse_part + all_rmse

  }
  # 计算残差
  residuals <- all_rmse/50

  return(residuals)
}

fit_and_calculate_rmse <- function(factors_df, dv_df, data_dv) {
  # 获取自变量的列名
  IDV_names <- colnames(factors_df)
  DV_names <- colnames(dv_df)

  # 线性回归
  if (length(DV_names)-1>0){
    formula_str_idv_1 <- paste(IDV_names, collapse = "+", sep = "+")
    formula_str_idv_2 <- paste(DV_names[1:(length(DV_names)-1)], collapse = "+", sep = "+")
    formula_str_idv <- paste(formula_str_idv_1, formula_str_idv_2, sep = "+")
  }else{
    formula_str_idv <- paste(IDV_names, collapse = "+", sep = "+")
  }
  formula_string <- paste(DV_names[length(DV_names)], "~", formula_str_idv)
  model <- lm(as.formula(formula_string), data = data_dv)
  #print(vif(model))
  print(anova(model,test = "Chisq"))
  print(summary(model))
  predicted_values <- as.numeric(predict(model, newdata = data_dv))
  diff_actual_values <- as.numeric(data_dv[, DV_names[length(DV_names)]])
  # 计算rmse并返回
  a <- (diff_actual_values - predicted_values)^2
  rmse <- sqrt(mean(a))
  return(list(rmse = rmse, model = model))
}

# 用于存储结果
result_df <- data.frame()
# 定义项目
project_name_ls <- as.list(unique(raw_data["project"]))$project
one_factors <- c("fork_entropy", "num_files", "num_forks", "project_age", "num_stars", "ratio_old_volunteers")
two_factors <- c("fork_entropy_rq2", "num_forks_rq2",  "ratio_old_volunteers_rq2",
                 "ratio_prs_with_tests","ratio_prs_with_hot_files", "num_closed_prs")
one_DV <- c("num_integrated_commits")
two_DV <- c("ratio_merged_prs")
three_DV <- c("num_bug_report_issues")


# 函数：执行一轮训练
# flag用于区别是否差分
# test_flag 用于区别是否测试
# current_flag用于区别是否只使用当前自变量的值，也即是标准AR模型
run_train <- function(raw_data, project_name_ls, factors, dv, len, flag=TRUE, current_flag=FALSE) {
  # 生成差分训练数据
  train_result <- generate_model_data(raw_data, project_name_ls, factors, dv, len, flag=flag, current_flag = current_flag, test_flag = FALSE)
  if(check_for_nan(train_result)){return (NaN)}
  factors_df <- train_result$factors_df
  dv_df <- train_result$dv_df
  pro_re_data <- train_result$pro_re_data

  # 拟合模型并计算RMSE
  rmse_and_model <- fit_and_calculate_rmse(factors_df, dv_df, pro_re_data)
  return(rmse_and_model)
}

# 函数：执行一轮测试
# flag用于区别是否差分, FALSE代表差分，TRUE代表原值
# test_flag 用于区别是否测试
# current_flag用于区别是否只使用当前自变量的值，也即是标准AR模型
run_test<- function(raw_data, project_name_ls, factors, dv, len, flag=TRUE, current_flag=FALSE) {
  #生成差分 测试数据，并将预测结果 与 真实结果进行比较
  test_model_data <- generate_test_model_data(raw_data, project_name_ls, factors, dv, len, current_flag = current_flag, test_flag = TRUE)
    # 合法性检查
  if (check_for_nan(test_model_data)) {
    return (NaN)
  }
  model <- test_model_data$test_model
  dv_df <- test_model_data$test_data$dv_df
  predicted_data <- test_model_data$test_data$pro_re_data
  # 测试
  DV_names <- colnames(dv_df)
  predicted_values <- predict(model, newdata = predicted_data)
  predicted_ <- data.frame(predicted_values = predicted_values, pro_name = predicted_data$pro_name)
  pros_included <- length(unique(predicted_$pro_name))

  # predicted_values <- as.numeric(predict(model, newdata = predicted_data))
  actual_values <- as.numeric(predicted_data[, DV_names[length(DV_names)]])
  # 如果TRUE的话表示进行预测差分还原
  if (!flag){
            # 这里必须是无差分的测试结果
            test_result_noDiff <- generate_model_data(raw_data, project_name_ls, factors, dv, len, flag = TRUE, current_flag = current_flag, test_flag = TRUE)
            true_value_df <- test_result_noDiff$pro_re_data
            # 恢复成原结果
            predicted_values_and_actual_values <- recover_original_timeseries(true_value_df, predicted_, DV_names[length(DV_names)])
            # 将项目传给predicted_values_and_actual_values
            predicted_values_and_actual_values["pro_name"] <- true_value_df["pro_name"]
            predicted_values <-  predicted_values_and_actual_values$handled_result
            actual_values <-  predicted_values_and_actual_values$actual_result
  }
  b <- (actual_values - predicted_values)^2
  rmse_predicted <- sqrt(mean(b))
  return(list(rmse_predicted=rmse_predicted,pros_included=pros_included))
}
max_index_Onedf <- data.frame(matrix(NA, nrow = 0, ncol = length(one_factors) * length(one_DV)))
# 主循环
rmse_trained_re <- matrix(NA, ncol = 3)
rmse_predicted_re <- matrix(NA, ncol = 4)
for (i in 1:1) {
  cat("Current len:", i, "\n")
  flag <- FALSE # 值约束
  current_flag <- FALSE # 自变量时移约束
  for (len in 11:11) {
    cat(len)
    project_name_ls_ <- project_name_ls

    rmse_and_model<- run_train(raw_data, project_name_ls_, one_factors, one_DV, len, flag = flag , current_flag=current_flag)
    one_rmse <- rmse_and_model$rmse
    one_rmse_predicted_and_pros <- run_test(raw_data, project_name_ls, one_factors, one_DV, len,  flag = flag  , current_flag=current_flag)
    one_rmse_predicted <- one_rmse_predicted_and_pros$rmse_predicted
    one_pros_included <- one_rmse_predicted_and_pros$pros_included

    # trmse_and_model<- run_train(raw_data, project_name_ls_, two_factors, two_DV, len, flag =flag , current_flag=current_flag)
    # two_rmse <- trmse_and_model$rmse
    # two_rmse_predicted_andpros<- run_test(raw_data, project_name_ls_, two_factors, two_DV, len,flag = flag , current_flag=current_flag)
    # two_rmse_predicted <- two_rmse_predicted_andpros$rmse_predicted
    # two_pros_included <- two_rmse_predicted_andpros$pros_included
    #
    # thrmse_and_model<- run_train(raw_data, project_name_ls_, one_factors, three_DV, len, flag =flag , current_flag=current_flag)
    # thr_rmse <-  thrmse_and_model$rmse
    # thr_rmse_predicted_andpros<- run_test(raw_data, project_name_ls_, one_factors, three_DV, len, flag =flag , current_flag=current_flag)
    # thr_rmse_predicted <- thr_rmse_predicted_andpros$rmse_predicted
    # thr_pros_included <- thr_rmse_predicted_andpros$pros_included
    #
    # rmse_trained_re <- rbind(rmse_trained_re, c(one_rmse ,two_rmse ,thr_rmse))
    # rmse_predicted_re <- rbind(rmse_predicted_re,c(one_rmse_predicted, two_rmse_predicted, thr_rmse_predicted,thr_pros_included))
}
}
# 将结果写入文件
if (!is.data.frame(rmse_trained_re)) {
    rmse_trained_re <- as.data.frame(rmse_trained_re)
}
if (!is.data.frame(rmse_predicted_re)) {
    rmse_predicted_re <- as.data.frame(rmse_predicted_re)
}
# write_xlsx(rmse_trained_re, path = "data/RmseResults_.xlsx")
# write_xlsx(rmse_predicted_re, path = "data/RmsePredictedResults_.xlsx")
# print("DONE!")