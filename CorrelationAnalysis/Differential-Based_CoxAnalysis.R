# 安装并加载必要的包
library("forecast")
library("dplyr")
library("tidyverse")    # needed for data manipulation
library("knitr")
library("stringr")
library("lme4")         # for the analysis
library("lmerTest")     # to get p-value estimations
library("performance")
library("car")
library("pscl")
library("sjstats")
library("emmeans")      # for interaction analysis
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
Sys.setlocale(category = 'LC_ALL', locale = 'English_United States.1252')

source("utils/generate_and_save_data.R")

setwd(".")

raw_data <- read.csv(
  "fork_entropy_dataset.csv",
  sep = ",",
  header = TRUE,
  encoding = "UTF-8",
  stringsAsFactors = FALSE)

generate_diff_ts <- function(raw_data, project_name, select){
  
  # 生成时间序列
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  
  # 设定百分位数的范围
  lower_percentile <- 0.5
  upper_percentile <- 99.5
  
  # 计算下界和上界的百分位数
  lower_bound <- quantile(orig_time_series, lower_percentile / 100)
  upper_bound <- quantile(orig_time_series, upper_percentile / 100)
  
  # 筛选在百分位数范围内的数据
  orig_time_series <- orig_time_series[orig_time_series >= lower_bound & orig_time_series <= upper_bound]
  
  # 一阶差分
  diff_series <- diff(orig_time_series, differences <- 1)
  
  # 中心化
  centered_diff_series <- diff_series - mean(diff_series)
  
  # 标准化
  standardized_diff_series <- centered_diff_series / sd(centered_diff_series)
  
  return(standardized_diff_series)
  
}


generate_lag_scatter_loop <- function(raw_data, project_name, select_factor, select_DV) {
  # 执行 generate_lag_scatter 方法
  max.lag <- 13
  con1 <- generate_diff_ts(raw_data, project_name, select_factor)
  series1 <- list(name = select_factor, con = con1)
  
  con2 <- generate_diff_ts(raw_data, project_name, select_DV)
  series2 <- list(name = select_DV, con = con2)
  
  # 加一个判断，如果series2皆为NaN则直接返回NaN,NaN
  if (all(is.nan(series2$con)) || all(is.nan(series1$con))) {
    return(c(NaN, NaN))
  }
  s1 <- as.ts(series1$con)
  s2 <- as.ts(series2$con)
  
  a <- ccf(s1, s2, max.lag, plot = FALSE)$acf[1:max.lag]
  h <- which.max(abs(a))
  max_value <- a[h]
  index_of_max <- 13 - h
  
  return (c(index_of_max,max_value))
}

select_columns <- c("fork_entropy","project_age","num_forks","num_files","num_stars","num_integrated_commits","ratio_merged_prs",
                   "num_bug_report_issues")
# 创建一个空数据框，用于存储结果
result_df <- data.frame()

# 定义项目
project_name_ls <- as.list(unique(raw_data["project"]))$project
# project_name_ls <- c("twbs/bootstrap","ansible/ansible","rails/rails")

one_factors <- c("num_forks","num_files", "fork_entropy","project_age","num_stars","ratio_old_volunteers")
two_factors <- c("num_forks_rq2","num_files_rq2", "fork_entropy_rq2","ratio_old_volunteers_rq2","ratio_prs_with_tests","ratio_prs_with_hot_files")

one_DV <- c("num_integrated_commits","num_bug_report_issues")
two_DV <- c("ratio_merged_prs")

# 以指定项目为例，画出所有的最大效应值对应的图示
# 循环遍历 select_columns 列表
max_index_Onedf <- data.frame(matrix(NA, nrow = 0, ncol = length(one_factors) * length(one_DV)))
for (project_name in project_name_ls){
  temp_re <- c()
  temp_name <- c()
  for (select_factor in one_factors) {
    for (select_DV in one_DV){
      index_and_value <- generate_lag_scatter_loop(raw_data, project_name, select_factor, select_DV)
      temp_re <- c(temp_re,index_and_value)
      temp_name <- c(temp_name,paste(select_factor,select_DV,"MAXindex",sep="-"))
      temp_name <- c(temp_name,paste(select_factor,select_DV,"MAXSlope",sep="-"))
    }
  }
  # 合并的一行是，
  max_index_Onedf <- rbind(max_index_Onedf, temp_re)
}
names(max_index_Onedf) <- temp_name

# 循环遍历 select_columns 列表
max_index_Twodf <- data.frame(matrix(NA, nrow = 0, ncol = length(two_factors) * length(two_DV)))
for (project_name in project_name_ls){
  temp_re <- c()
  temp_name <- c()
  for (select_factor in two_factors) {
    for (select_DV in two_DV) { 
      # 执行 generate_lag_scatter_loop 方法
      index_and_value <- generate_lag_scatter_loop(raw_data, project_name, select_factor, select_DV)
      temp_re <- c(temp_re,index_and_value)
      temp_name <- c(temp_name,paste(select_factor,select_DV,"MAXindex",sep="-"))
      temp_name <- c(temp_name,paste(select_factor,select_DV,"MAXSlope",sep="-"))
    }
  }
  max_index_Twodf <- rbind(max_index_Twodf, temp_re)
}
names(max_index_Twodf) <- temp_name
max_index_df <- cbind(max_index_Onedf, max_index_Twodf)
#max_index_df <- replace(max_index_df, is.na(max_index_df), 0)

file_path <- "data/EA_Difference_chapter2_ALL50Pro_ALLVar.csv"
# 将 result_df 保存为CSV文件
write.csv(max_index_df, file <- file_path, row.names <- FALSE)

