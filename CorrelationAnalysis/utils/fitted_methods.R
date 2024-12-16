
arma_fit_and_calculate_rmse <- function(factors_df, dv_df, data_dv, original_data_dv, arma_order=c(1,1,0)) {
  # 获取自变量的列名
  IDV_names <- colnames(factors_df)
  DV_names <- colnames(dv_df)
  
  # 使用ARIMA模型
  arma_formula <- paste(DV_names[length(DV_names)], "~", paste(IDV_names, collapse = "+", sep = "+"))
  arma_model <- arima(original_data_dv[, DV_names[length(DV_names)]], order = arma_order)
  
  # 利用模型进行预测
  predicted_values <- as.numeric(predict(arma_model, n.ahead = nrow(dv_df))$pred)
  
  # 输出模型概要
  print(summary(arma_model))
  
  # 恢复成原结果
  predicted_values = recover_original_timeseries(original_data_dv, predicted_values, DV_names[length(DV_names)])
  
  # 真实结果集
  actual_values <- as.numeric(original_data_dv[, DV_names[length(DV_names)]])
  
  # 计算残差
  residuals_values <- actual_values - predicted_values
  
  # 计算 RMSE
  rmse <- sqrt(mean(residuals_values^2, na.rm = TRUE))
  
  return(rmse)
}

fit_model <- function(train_result) {
  factors_df <- train_result$factors_df
  dv_df <- train_result$dv_df
  data_dv <- train_result$pro_re_data
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
  return(model)
}