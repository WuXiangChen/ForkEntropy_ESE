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
library("writexl")
source("./utils/generate_and_save_data.R")

setwd("./")

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

perform_whitening_and_test <- function(orig_time_series, lag = 20) {
  # 差分
  diff_series <- orig_time_series
  
  # 标准化
  scaled_data <- scale(diff_series)
  
  # 计算协方差矩阵
  cov_matrix <- cov(scaled_data)
  
  # 特征值分解
  eigen_decomp <- eigen(cov_matrix)
  
  # 白化变换
  whitening_matrix <- eigen_decomp$vectors %*% diag(1/sqrt(eigen_decomp$values))
  whitened_data <- scaled_data %*% whitening_matrix
  
  # Box-Pierce 检验（未白化）
  box_pierce_test_before <- Box.test(diff_series, lag = lag, type = "Box-Pierce")
  cat("Box-Pierce Test (before whitening):\n")
  print(box_pierce_test_before)
  
  # Box-Pierce 检验（白化后）
  box_pierce_test_after <- Box.test(whitened_data, lag = lag, type = "Box-Pierce")
  cat("\nBox-Pierce Test (after whitening):\n")
  print(box_pierce_test_after)
  
  return (whitened_data)
}

time_series_analysis <- function(raw_data, project_name, select) {
  
  print(select)
  # 生成时间序列
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)),  frequency = 12)
  # 作预筛选
  
  # 设定百分位数的范围
  lower_percentile <- 0.5
  upper_percentile <- 99.5
  
  # 计算下界和上界的百分位数
  lower_bound <- quantile(orig_time_series, lower_percentile / 100)
  upper_bound <- quantile(orig_time_series, upper_percentile / 100)
  
  # 筛选在百分位数范围内的数据
  orig_time_series <- orig_time_series[orig_time_series >= lower_bound & orig_time_series <= upper_bound]
  
  
  #orig_time_series = scale(orig_time_series)
  
  # 对一阶差分后的序列进行一阶差分
  diff_series <- diff(orig_time_series)
  
  # 进行ADF检验
  result_adf <- adf.test(orig_time_series, k=4)
  adf_statistic_ori <- result_adf$statistic
  adf_p_value_ori <- result_adf$p.value
  
  # 进行ADF检验（差分后的序列）
  result_adf_diff <- adf.test(diff_series, k=4)
  adf_statistic_diff <- result_adf_diff$statistic
  adf_p_value_diff <- result_adf_diff$p.value
  
  
  # 进行KPSS检验
  result_kpss <- kpss.test(orig_time_series, lshort = TRUE, null="Trend")
  kpss_statistic_ori <- result_kpss$statistic
  kpss_p_value_ori <- result_kpss$p.value
  
  # 进行KPSS检验（差分后的序列）
  result_kpss_diff <- kpss.test(diff_series, lshort = TRUE,null="Trend")
  kpss_statistic_diff <- result_kpss_diff$statistic
  kpss_p_value_diff <- result_kpss_diff$p.value
  
  # 进行Ljung-Box检验
  ljung_box_test <- Box.test(orig_time_series, lag = 20, type = "Ljung-Box")
  ljung_box_statistic_ori <- ljung_box_test$statistic
  ljung_box_p_value_ori <- ljung_box_test$p.value
  
  # 使用 Box.test 进行 Box-Pierce 检验
  box_pierce_test <- Box.test(diff_series, lag = 20, type = "Ljung-Box")
  box_pierce_statistic_diff <- box_pierce_test$statistic
  box_pierce_p_value_diff <- box_pierce_test$p.value
  
  # 使用示例
  perform_whitening_and_test(orig_time_series, lag = 20)
  
  
  # 返回结果
  result <- list(
    ADF_Statistic_Original_Series = adf_statistic_ori,
    ADF_P_Value_Original_Series = adf_p_value_ori,
    ADF_Statistic_Diff_Series = adf_statistic_diff,
    ADF_P_Value_Diff_Series = adf_p_value_diff,
    
    KPSS_Statistic_Original_Series = kpss_statistic_ori,
    KPSS_P_Value_Original_Series = kpss_p_value_ori,
    KPSS_Statistic_Diff_Series = kpss_statistic_diff,
    KPSS_P_Value_Diff_Series = kpss_p_value_diff,
    
    Ljung_Box_Statistic_Original_Series = ljung_box_statistic_ori,
    Ljung_Box_P_Value_Original_Series = ljung_box_p_value_ori,
    Box_Pierce_Statistic_Diff_Series = box_pierce_statistic_diff,
    Box_Pierce_P_Value_Diff_Series = box_pierce_p_value_diff
  )
  return(result)
}

generate_diff_ts <- function(raw_data, project_name, select){
  
  # 生成时间序列
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  
  # 设定百分位数的范围
  lower_percentile <- 0.05
  upper_percentile <- 99.95
  
  # 计算下界和上界的百分位数
  lower_bound <- quantile(orig_time_series, lower_percentile / 100)
  upper_bound <- quantile(orig_time_series, upper_percentile / 100)
  
  # 筛选在百分位数范围内的数据
  orig_time_series <- orig_time_series[orig_time_series >= lower_bound & orig_time_series <= upper_bound]
  # 一阶差分
  diff_series <- diff(orig_time_series, differences = 1)
  
  return (diff_series)
  
}

get_ori_ts <- function(raw_data, project_name, select){
  
  # 生成时间序列
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  
  # 设定百分位数的范围
  lower_percentile <- 0.05
  upper_percentile <- 99.95
  
  # 计算下界和上界的百分位数
  lower_bound <- quantile(orig_time_series, lower_percentile / 100)
  upper_bound <- quantile(orig_time_series, upper_percentile / 100)
  
  # 筛选在百分位数范围内的数据
  orig_time_series <- orig_time_series[orig_time_series >= lower_bound & orig_time_series <= upper_bound]
  # 一阶差分
  diff_series <- diff(orig_time_series, differences = 1)
  
  return (orig_time_series)
  
}

generate_lag_scatter <- function(series1, series2){
  
  # 绘图参数
  col <- gray(0.1)
  cex <- 0.9
  ltcol <- 1
  box.col <- 8
  lwl <- 1
  lwc <- 2
  bgl <- gray(1, 0.65)
  max.lag <- 12
  
  
  # 进行等长去尾值
  length_diff <- length(series1$con) - length(series2$con)
  
  # 判断是否需要去尾值
  if (length_diff > 0) {
    # series1较长，去掉series1的尾部值，使其长度与series2相等
    series1$con <- head(series1$con, length(series2$con))
  } else if (length_diff < 0) {
    # series2较长，去掉series2的尾部值，使其长度与series1相等
    series2$con <- head(series2$con, length(series1$con))
  }
  
  s1 <- as.ts(series1$con)
  s2 <- as.ts(series2$con)
  
  # 取最值
  a <- ccf(s1, s2, max.lag, plot = FALSE)$acf[1:max.lag]
  h <- which.max(abs(a))
  
  lagNum <- h-13
  u <- ts.intersect(stats::lag(s1, lagNum), s2)
  
  Xlab <- paste(series1$name,"(t", lagNum, ")", sep = "")
  tsplot(u[, 1], u[, 2], type = "p", xy.labels = FALSE, 
         xy.lines = FALSE, xlab = Xlab, ylab = series2$name, col = col, cex = cex)
  
  # 绘制平滑曲线
  lines(stats::lowess(u[, 1], u[, 2]), col = lwc,lwd = lwl)
  
  # 创建图例
  legend("topright", legend = format(round(a[h], digits = 2), nsmall = 2), text.col = ltcol, 
         bg = bgl, adj = 0.25, box.col = box.col, cex = 0.9)
}

select_columns <- c("num_forks","num_files","fork_entropy","project_age","num_stars",
                    "ratio_old_volunteers", "ratio_prs_with_tests", "ratio_prs_with_hot_files",
                    "num_integrated_commits","ratio_merged_prs", "num_bug_report_issues")
result_df <- data.frame()

# 定义项目
project_name <- "angular/angular.js"

# 循环遍历 select_columns 列表
for (select_column_name in select_columns) {
  # 执行 time_series_analysis 方法
  result <- time_series_analysis(raw_data, project_name, select_column_name)

  # 将结果添加到数据框中
  result_df <- rbind(result_df, data.frame(
    Column <- select_column_name,
    ADF_Dickey_Fuller_Original_Series <- result$ADF_Statistic_Original_Series,
    ADF_P_Value_Original_Series <- result$ADF_P_Value_Original_Series,
    ADF_Dickey_Fuller_Diff_Series <- result$ADF_Statistic_Diff_Series,
    ADF_P_Value_Diff_Series <- result$ADF_P_Value_Diff_Series,

    KPSS_KPSS_Level_Original_Series <- result$KPSS_Statistic_Original_Series,
    KPSS_P_Value_Original_Series <- result$KPSS_P_Value_Original_Series,
    KPSS_KPSS_Level_Diff_Series <- result$KPSS_Statistic_Diff_Series,
    KPSS_P_Value_Diff_Series <- result$KPSS_P_Value_Diff_Series,

    Box_Ljung_chisq_Original_Series <- result$Ljung_Box_Statistic_Original_Series,
    Box_Ljung_P_Value_Original_Series <- result$Ljung_Box_P_Value_Original_Series,
    Box_Ljung_chisq_Diff_Series <- result$Box_Pierce_Statistic_Diff_Series,
    Box_Ljung_P_Value_Diff_Series <- result$Box_Pierce_P_Value_Diff_Series
  ))
}
# 将 验证test结果 保存为CSV文件
file_path <- "data/chapter2_MultiTest_.xlsx"
write_xlsx(result_df, path = file_path)



select_columns <- c("fork_entropy")
# 创建一个空数据框，用于存储结果
result_df <- data.frame()
forkEntropyR <- generate_diff_ts(raw_data, project_name, "fork_entropy")
whiten_forkEntropyR <- perform_whitening_and_test(forkEntropyR)
ori_forkEntropyR <- get_ori_ts(raw_data, project_name, "fork_entropy_rq2")

pacf(ori_forkEntropyR,lag.max=12)

ori_forkEntropyR.acfout <- acf(ori_forkEntropyR,plot=FALSE,lag.max=11)$acf
ori_forkEntropyR.pacfout <- pacf(ori_forkEntropyR,plot=FALSE,lag.max=12)$acf

whiten_forkEntropyR.acfout <- acf(whiten_forkEntropyR,plot=FALSE,lag.max=11)$acf
whiten_forkEntropyR.pacfout <- pacf(whiten_forkEntropyR,plot=FALSE,lag.max=12)$acf

whiten_test_df <- data.frame(
  ori_forkEntropyR_acf = unlist(ori_forkEntropyR.acfout),
  ori_forkEntropyR_pacf = unlist(ori_forkEntropyR.pacfout),
  whiten_forkEntropyR_acf = unlist(whiten_forkEntropyR.acfout),
  whiten_forkEntropyR_pacf = unlist(whiten_forkEntropyR.pacfout)
)

# 打印或进一步处理 result_df 数据框
print(whiten_test_df)

# 将 whiten_test_df 保存为CSV文件
file_path <- "data/chapter2_whiten_test_df_.csv"
write.csv(whiten_test_df, file = file_path, row.names = FALSE)


# 绘制差分后的时间序列图和ACF图
# par(mfrow = c(3,1))
# plot(your_time_series, main = "Original Time Series")
# # plot(diff_series_1, main = "First Differenced Time Series Plot")
# acf_result_ori <- acf(your_time_series, main = "ACF Plot of Original Time Series", alpha = 0.05)
# # 添加图例
# legend("topright", legend = "95% Confidence Limits", lty = 2, col = "blue", bty = "n")
# # plot(diff_series_2, main = "Second Differenced Time Series Plot")
# acf_result_diff <- acf(diff_series_1, main = "ACF Plot of First Differenced Series", alpha = 0.05)
# legend("topright", legend = "95% Confidence Limits", lty = 2, col = "blue", bty = "n")