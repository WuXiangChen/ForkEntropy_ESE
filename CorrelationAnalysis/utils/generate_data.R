
split_ratio = 0.8
generate_train_ts <- function(raw_data, project_name, select, flag=FALSE){
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  centered_series <- orig_time_series - mean(orig_time_series)
  standardized_series <- centered_series / (sd(orig_time_series)+0.00001)
  if (flag) {
    training_length <- round(split_ratio * length(standardized_series))
    training_data <- head(standardized_series, training_length)
  } else {
    diff_series <- diff(standardized_series, differences = 1)
    training_length <- round(split_ratio * length(diff_series))
    training_data <- head(diff_series, training_length)
  }
  return(training_data)
}

generate_test_ts <- function(raw_data, project_name, select, flag=FALSE){
  orig_time_series <- ts(subset(raw_data, project == project_name, select = c(select)))
  centered_series <- orig_time_series - mean(orig_time_series)
  standardized_series <- centered_series / (sd(orig_time_series)+0.00001)

  if (flag) {
    testing_length <- round(split_ratio * length(standardized_series))
    testing_data <- tail(standardized_series,length(standardized_series) - testing_length)
  } else {
    diff_series <- diff(standardized_series, differences = 1)
    testing_length <- round(split_ratio * length(diff_series))
    testing_data <- tail(diff_series, length(diff_series) - testing_length)
  }
  return(testing_data)
}

generate_lag_data_for_fitting <- function(raw_data, project_name, select_factor, k, flag=FALSE, current_flag=FALSE,test_flag=FALSE) {
  if (test_flag){
      con1 <- as.numeric(generate_test_ts(raw_data, project_name, select_factor, flag=flag))
  }else{
      con1 <- as.numeric(generate_train_ts(raw_data, project_name, select_factor, flag=flag))
  }
  features_df <- data.frame(select_factor = con1[1:(length(con1)-k+1)])
  colnames(features_df)[1] <- select_factor
  if (k<2){
    return (features_df)
  }
  for (i in 2:k){
    tmp <- con1[i:(length(con1)-(k-i))]
    lag_name <- paste(select_factor,i-1,sep = "_")
    features_df$lag_name <- tmp
    colnames(features_df)[i] <- lag_name
  }

   features_df[is.na(features_df)] <- 0

  if (current_flag){
    return  (features_df[lag_name])
  }
  return (features_df)
}

generate_lag_dv_for_fitting <- function(raw_data, project_name, select_DV, k, flag=FALSE) {
  con1 <- as.numeric(generate_train_ts(raw_data, project_name, select_DV, flag))
  tmp <- con1[(k+1):(length(con1))]
  
  lag_name = paste(select_DV)
  features_df = data.frame(lag_name = tmp)
  colnames(features_df)[1] <- lag_name
  
  return (features_df)
}

generate_test_model_data <- function(raw_data, project_name_ls, factors, DV, len, flag=FALSE , test_flag=FALSE, current_flag=FALSE) {
  test_model <- getfited_model(raw_data, project_name_ls, factors, DV, len,flag=flag, current_flag = current_flag,test_flag = FALSE)
  test_data <- generate_model_data(raw_data, project_name_ls, factors, DV, len, flag=flag, current_flag=current_flag,test_flag=TRUE)
  return (list(test_model=test_model, test_data=test_data))
}

generate_model_data <- function(raw_data, project_name_ls, factors, DV, len, flag=TRUE , test_flag=FALSE, current_flag=FALSE) {
  pro_re_data <- data.frame()
  for (project_name in project_name_ls) {
    # print(project_name)
    if (test_flag){
      len_test_data <- generate_test_ts(raw_data, project_name, factors[1], flag=flag)
    }else{
      len_test_data <- generate_train_ts(raw_data, project_name, factors[1], flag=flag)
    }
    if (nrow(len_test_data) <= len) {
        next
    }
    factors_df <- data.frame()
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
    
    data_dv <- cbind(factors_df, dv_df)
    pro_name <- rep(c(project_name), nrow(dv_df))
    data_dv$pro_name <- pro_name
    pro_re_data <- rbind(pro_re_data, data_dv)
  }
  return(list(factors_df = factors_df, dv_df = dv_df, pro_re_data = pro_re_data))
}

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