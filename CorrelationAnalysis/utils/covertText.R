convert_text <- function(input_text) {
  # 使用 gsub 将下划线后的单词首字母大写
  result <- gsub("^(.)?", "\\U\\1", input_text, perl = TRUE)
  result <- gsub("_(.)", "_\\U\\1", result, perl = TRUE)
  result <- gsub("(_Rq2)", "", result, perl = TRUE)
  # 去除下划线
  result <- gsub("_", "", result)
  
  return(result)
}
