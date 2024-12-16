#生成对应的数据
split_ratio = 0.8
# 这边更加准确的来说，应该是生成训练数据
generate_train_ts <- function(raw_data, project_name, select, flag=FALSE){
  # 生成时间序列
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  # 判断 project_name 是否以 "fork_entropy" 开头
  if (startsWith(select, "fork_entropy")) {
      # 使用对数变换进行归一化
      standardized_series <- log(orig_time_series + 0.01)
  } else {
      # 使用 Z-score 标准化
      std_dev <- sd(orig_time_series)
      standardized_series <- (orig_time_series - mean(orig_time_series)) / (std_dev + 1e-10)
  }

  if (flag) {
    # 这里输出给定指定长度的训练数据
    training_length <- round(split_ratio * length(standardized_series))
    # 提取前80%的训练数据
    training_data <- head(standardized_series, training_length)
  } else {
    # 一阶差分
    diff_series <- diff(standardized_series, differences = 1)
    # 这里输出给定指定长度的训练数据
    training_length <- round(split_ratio * length(diff_series))
    # 提取前80%的训练数据
    training_data <- head(diff_series, training_length)
  }
  return(training_data)
}

#生成测试数据
generate_test_ts <- function(raw_data, project_name, select, flag=FALSE){
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  standardized_series <- log(orig_time_series + 0.01)
  mean_output <- sprintf("mean(orig_time_series): %.2f, %s", mean(orig_time_series),select)
  # print(mean_output)

  std_output <- sprintf("std(orig_time_series): %.2f,%s", sd(orig_time_series),select)
  # print(std_output)

  # standardized_series <- orig_time_series
  if (flag) {
    # 这里输出给定指定长度的训练数据
    testing_length <- round(split_ratio * length(standardized_series))
    # 提取前80%的训练数据
    testing_data <- tail(standardized_series,length(standardized_series) - testing_length)
  } else {
    # 一阶差分
    diff_series <- diff(standardized_series, differences = 1)
    # 这里输出给定指定长度的训练数据
    testing_length <- round(split_ratio * length(diff_series))
    # 提取前80%的训练数据
    testing_data <- tryCatch({
      tail(diff_series, length(diff_series) - testing_length)
    }, warning = function(w) {
      message("警告: ", conditionMessage(w))
      return(NULL)  # 返回NULL或其他值以表示警告情况
    }, error = function(e) {
      message("错误: ", conditionMessage(e))
      return(NULL)  # 返回NULL或其他值以表示错误情况
    })
  }
  return(testing_data)
}
# flag用于区别是否差分
# test_flag 用于区别是否测试
# current_flag用于区别是否只使用当前自变量的值，也即是标准AR模型
generate_lag_data_for_fitting <- function(raw_data, project_name, select_factor, k, flag=FALSE, current_flag=FALSE,test_flag=FALSE) {
  # 参数k为最大滞后值
  if (test_flag){
      con1 <- as.numeric(generate_test_ts(raw_data, project_name, select_factor, flag=flag))
  }else{
      con1 <- as.numeric(generate_train_ts(raw_data, project_name, select_factor, flag=flag))
  }
  # 这里用于获取延迟数据
  features_df <- data.frame(select_factor = con1[1:(length(con1)-k+1)])
  colnames(features_df)[1] <- select_factor
  # 现在需要返回的是 直至最大滞后的所有时序结果
  if (k<2){
    return (features_df)
  }
  for (i in 2:k){
    tmp <- con1[i:(length(con1)-(k-i))]
    lag_name <- paste(select_factor,i-1,sep = "_")
    features_df$lag_name <- tmp
    colnames(features_df)[i] <- lag_name
  }
   # 这里生成数据是没问题的，并且这里没有用Null来补完
   features_df[is.na(features_df)] <- 0
  if (current_flag){
    return  (features_df[lag_name])
  }
  return (features_df)
}

generate_lag_dv_for_fitting <- function(raw_data, project_name, select_DV, k, flag=FALSE) {
  # 参数k为最大滞后值
  con1 <- as.numeric(generate_train_ts(raw_data, project_name, select_DV, flag))
  tmp <- con1[(k+1):(length(con1))]
  
  lag_name <- paste(select_DV)
  features_df <- data.frame(lag_name = tmp)
  colnames(features_df)[1] <- lag_name
  return (features_df)
}

generate_test_model_data <- function(raw_data, project_name_ls, factors, DV, len, flag=FALSE , test_flag=FALSE, current_flag=FALSE) {
  train_result <- generate_model_data(raw_data, project_name_ls, factors, DV, len,flag=flag, current_flag = current_flag,test_flag = FALSE)
  test_model <- fit_model(train_result)
  test_data <- generate_model_data(raw_data, project_name_ls, factors, DV, len, flag=flag, current_flag=current_flag,test_flag=TRUE)
  return (list(test_model=test_model, test_data=test_data))
}

generate_model_data <- function(raw_data, project_name_ls, factors, DV, len, flag=TRUE , test_flag=FALSE, current_flag=FALSE) {
  pro_re_data <- data.frame()
  for (project_name in project_name_ls) {
    if (test_flag){
      len_test_data <- generate_test_ts(raw_data, project_name, factors[1], flag=flag)
    }else{
      len_test_data <- generate_train_ts(raw_data, project_name, factors[1], flag=flag)
    }
    if (nrow(len_test_data) <= len) {
        next
    }
    factors_df <- data.frame()
    # 获取 自变量时移信息
    for (select_factor in factors) {
      # print(select_factor)
      df_no_na <- generate_lag_data_for_fitting(raw_data, project_name, select_factor, len, flag=flag, test_flag=test_flag,current_flag=current_flag)
      if (nrow(factors_df) == 0) {
        factors_df <- df_no_na
      } else {
        factors_df <- cbind(factors_df, df_no_na)
      }
    }
    dv_df <- data.frame()
    for (select_DV in DV) {
      df_no_na <- generate_lag_data_for_fitting(raw_data, project_name, select_DV, len, flag=flag, test_flag=test_flag)
      if (nrow(dv_df) == 0) {
        dv_df <- df_no_na
      } else {
        dv_df <- cbind(dv_df, df_no_na)
      }
    }
    
    data_dv  <- cbind(factors_df, dv_df)
    pro_name <- rep(c(project_name), nrow(dv_df))
    platform <- subset(raw_data, project == project_name, select = c(2))[1,1]
    platform <- rep(c(platform), nrow(dv_df))
    data_dv$pro_name <- pro_name
    data_dv$platform <- platform
    pro_re_data <- rbind(pro_re_data, data_dv)
  }
  return(list(factors_df = factors_df, dv_df = dv_df, pro_re_data = pro_re_data))
}



# 这是标准的AR模型，备以测试
generate_model_data_OnlyCurrent_demo <- function(raw_data, project_name_ls, factors, DV, len, flag=FALSE,test_flag=FALSE) {
  pro_re_data <- data.frame()
  for (project_name in project_name_ls) {
    factors_df <- data.frame()
    for (select_factor in factors) {
      df_no_na <- generate_lag_data_for_fitting(raw_data, project_name, select_factor, len, flag=flag,current_flag=TRUE,test_flag=test_flag)
      if (nrow(factors_df) == 0) {
        factors_df <- df_no_na
      } else {
        factors_df <- cbind(factors_df, df_no_na)
      }
    }

    dv_df <- data.frame()
    for (select_DV in DV) {
      df_no_na <- generate_lag_data_for_fitting(raw_data, project_name, select_DV, len, flag,test_flag=test_flag)
      if (nrow(dv_df) == 0) {
        dv_df <- df_no_na
      } else {
        dv_df <- cbind(dv_df, df_no_na)
      }
    }
    data_dv <- cbind(factors_df, dv_df)
    pro_name <- rep(c(project_name), nrow(dv_df))
    data_dv$pro_name <- pro_name
    pro_re_data <- rbind(pro_re_data, data_dv)
  }
  return(list(factors_df = factors_df, dv_df = dv_df, pro_re_data = pro_re_data))
}