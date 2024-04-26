is_valid_data <- function(predicted_data) {
  num_rows <- nrow(predicted_data)
  return(num_rows > 0)
}

check_for_nan <- function(data_vector) {
  if (!is.list(data_vector)) {
    return(TRUE)
  } else {
    return (FALSE)
  }
}