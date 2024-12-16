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

source("utils/generate_and_save_data.R")
source("utils/covertText.R")

setwd("./")

raw_data <- read.csv(
  "fork_entropy_dataset.csv",
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
  # 中心化
  centered_diff_series <- diff_series - mean(diff_series)

  # 标准化
  standardized_diff_series <- centered_diff_series / sd(centered_diff_series)


  return(standardized_diff_series)

}

generate_lag_scatter_loop <- function(raw_data, project_name, select_factor, select_DV) {

  max.lag = 12
  # 执行 generate_lag_scatter 方法
  con1 <- generate_diff_ts(raw_data, project_name, select_factor)
  series1 <- list(name = select_factor, con = con1)

  con2 <- generate_diff_ts(raw_data, project_name, select_DV)
  series2 <- list(name = select_DV, con = con2)
  s1 = as.ts(series1$con)
  s2 = as.ts(series2$con)
  a = ccf(s1, s2, max.lag, plot = FALSE)$acf[1:max.lag]
  h =  which.max(abs(a))
  return (h)
}

select_columns = c("fork_entropy","project_age","num_forks","num_files","num_stars","num_integrated_commits","ratio_merged_prs",
                   "num_bug_report_issues")
# 创建一个空数据框，用于存储结果
result_df <- data.frame()
# 定义项目
project_name_ls <- c("twbs/bootstrap","ansible/ansible","rails/rails")
one_factors = c("num_forks","num_files", "fork_entropy")
two_factors = c("num_forks_rq2","num_files_rq2", "fork_entropy_rq2")

one_DV = c("num_integrated_commits","num_bug_report_issues")
two_DV = c("ratio_merged_prs")

max_index_Onedf <- data.frame(matrix(NA, nrow = 0, ncol = length(one_factors) * length(one_DV)))
for (project_name in project_name_ls){
  temp_re = c()
  # 控制绘图布局
  # par(mfrow = c(3, 4))
  for (select_factor in one_factors) {
    for (select_DV in one_DV){
      # 执行 generate_lag_scatter_loop 方法
      index_of_max <- generate_lag_scatter_loop(raw_data, project_name, select_factor, select_DV)
      temp_re <- c(temp_re,index_of_max)
    }
  }

  max_index_Onedf <- rbind(max_index_Onedf, temp_re)
}
tmp <- paste(one_factors, c("num_integrated_commits"), sep="-")
tmp_ <- paste(one_factors, c("num_bug_report_issues"), sep="-")
column_name = c(tmp, tmp_)
colnames(max_index_Onedf) <- column_name


# 循环遍历 select_columns 列表
max_index_Twodf <- data.frame(matrix(NA, nrow = 0, ncol = length(two_factors) * length(two_DV)))
for (project_name in project_name_ls){
  temp_re = c()
  for (select_factor in two_factors) {
    for (select_DV in two_DV) {
      # 执行 generate_lag_scatter_loop 方法
      index_of_max <- generate_lag_scatter_loop(raw_data, project_name, select_factor, select_DV)
      temp_re <- c(temp_re,index_of_max)
    }
  }
  max_index_Twodf <- rbind(max_index_Twodf, temp_re)
}

colnames(max_index_Twodf) <- rep(paste(two_factors, one_DV, sep="-"), each = length(two_DV))

max_index_df = cbind(max_index_Onedf, max_index_Twodf)

plots_data = c()
# 使用 for 循环遍历列
for (col_name in names(max_index_df)) {
  factor <- strsplit(col_name,'-')[[1]][1]
  dv <- strsplit(col_name,'-')[[1]][2]
  col_values <- max_index_df[[col_name]]
  for (i in seq_along(project_name_ls)) {
    project_name <- project_name_ls[i]
    con1 <- generate_diff_ts(raw_data, project_name, factor)
    con2 <- generate_diff_ts(raw_data, project_name, dv)

    s1 = as.ts(con1)
    s2 = as.ts(con2)

    u = ts.intersect(s1, s2)
    column_name_factor = paste(project_name,factor,dv,"1",sep="-")
    column_name_dv = paste(project_name,factor,dv,"2",sep="-")
    tmp = data.frame(column_name = u)
    names(tmp) <- c(column_name_factor, column_name_dv)
    plots_data <- c(plots_data, tmp)
  }
}

par(mfrow = c(1, 1))
par(mar=c(4.5,0.5,0,0.5),oma=c(0,0,0,0))
windowsFonts("Arial" = windowsFont("Arial"))
par(family = "Arial")

lagdf_value_names = names(max_index_df)
max.lag = 12
lwd = 4
cex = 4
# 每次更改oneDv 这类参数时，重新控制循环轮次
for (c_part in list(c(1,4),c(2,3,5,6),c(7),c(8,9))){
  for (i in c_part){
  lag_value = max_index_df[[lagdf_value_names[i]]]
  # 绘制第一个项目的内容
  factor = plots_data[(i-1)*6+1]
  column_names <- names(factor)[1]
  factor <- as.numeric(factor[[column_names]])
  dv = plots_data[(i-1)*6+2]
  column_names <- names(dv)[1]
  dv <- as.numeric(dv[[column_names]])

  xlab <- NA
  ylab <- NA

  # 添加网格示例
  plot(factor, dv, type = "p", lwd = 2,  col = "blue", axes=FALSE,
       xlab=xlab, ylab=ylab, font.axis = 1,
       xlim = c(-3, 3.5), ylim = c(-3, 3.5))
  box()
  grid(col = "gray", lty = 1)
  axis(side=1,at=seq(-3, 3.5,1), tck = 0,tick=TRUE, cex.axis=5,font=1,mgp=c(0,3,0))
  # axis(side=2,at=seq(-3, 3.5,1), tck = 1,tick=TRUE,gap.axis=1, cex.axis=5,font=1)

  linear_model <- lm(dv ~ factor)
  # 这里在计算互相关的值
  cox1 = ccf(factor, dv, max.lag, plot = FALSE)$acf[1:max.lag][lag_value[1]]
  lines(stats::lowess(factor, dv) ,lwd = lwd, col = "blue")

  x = 2
  slope1 = coef(linear_model)[2]
  y = slope1*x + round(coef(linear_model)[1],3) + 0.6
  lg_ = paste("t=",12-lag_value[1],sep = "")
  slope_ = paste("xcorr=",sprintf("%.3f", cox1),sep = "")
  lg1 = paste("(",slope_,"，",lg_,")")
  srt_angle = atan(slope1) * (180 / pi)
  # 在直线上标注数字
  text(x = x, y = y, labels = lg1, pos = 1, col = "blue",
       srt = srt_angle, cex = cex, font = 2)

  factor = plots_data[(i-1)*6+3]
  column_names <- names(factor)[1]
  factor <- as.numeric(factor[[column_names]])

  dv = plots_data[(i-1)*6+4]
  column_names <- names(dv)[1]
  dv <- as.numeric(dv[[column_names]])
  points(factor, dv, pch = 6,lwd = 2, col = "orange")
  linear_model <- lm(dv ~ factor)

  # 这里在计算互相关的值
  cox2 = ccf(factor, dv, max.lag, plot = FALSE)$acf[1:max.lag][lag_value[2]]
  lines(stats::lowess(factor, dv),lwd = lwd,  col = "orange")

  x = 1.0
  slope2 <- coef(linear_model)[2]
  y = slope2*x + round(coef(linear_model)[1],3)-0.4
  lg_ = paste("t=",12-lag_value[2],sep = "")
  slope_ = paste("xcorr=",sprintf("%.3f", cox2),sep = "")
  lg2 = paste("(",slope_,"，",lg_,")")
  srt_angle = 0
  # 在直线上标注数字
  text(x = x, y = y, labels = lg2, pos = 1, col = "orange",
       srt = srt_angle, cex = cex, font = 2)

  factor = plots_data[(i-1)*6+5]
  column_names <- names(factor)[1]
  factor <- as.numeric(factor[[column_names]])

  dv = plots_data[(i-1)*6+6]
  column_names <- names(dv)[1]
  dv <- as.numeric(dv[[column_names]])
  points(factor, dv, pch = 4,lwd = 2, col = "red")
  linear_model <- lm(dv ~ factor)
  # abline(linear_model, col = "red", lwd = 1)

  # 这里在计算互相关的值
  cox3 = ccf(factor, dv, max.lag, plot = FALSE)$acf[1:max.lag][lag_value[3]]
  lines(stats::lowess(factor, dv),lwd = lwd,col = "red")
  x = -1.5
  slope3 <- coef(linear_model)[2]
  y = slope3*x + round(coef(linear_model)[1],3) + 0.5
  lg_ = paste("t=",12-lag_value[3],sep = "")
  slope_ = paste("xcorr=",sprintf("%.3f", cox3),sep = "")
  lg3 = paste("(",slope_,"，",lg_,")")
  srt_angle = atan(slope3) * (180 / pi)
  # 在直线上标注数字
  text(x = x, y = y, labels = lg3, pos = 1, col = "red",
       srt = srt_angle, cex = cex, font = 2)
  }
}
print(length(plots_data))