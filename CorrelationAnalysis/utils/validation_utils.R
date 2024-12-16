is_valid_data <- function(predicted_data) {
  # 获取 predicted_data 的行数
  num_rows <- nrow(predicted_data)

  # 返回是否为合法数据的逻辑值
  return(num_rows > 0)
}

check_for_nan <- function(data_vector) {
  if (!is.list(data_vector)) {
    # 数据中存在 NaN 值的处理代码
    return(TRUE)
  } else {
    # 数据中没有 NaN 值的处理代码
    return (FALSE)
  }
}