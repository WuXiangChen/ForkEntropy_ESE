
library("data.table")  # 用于数据表操作
library("dplyr")
library("tidyverse")    # needed for data manipulation
library("knitr")
library("stringr")
library("lme4")         # for the analysis
library("lmerTest")     # to get p-value estimations
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

generate_and_save_emmip <- function(rq_model,formula, file,vary_project_age_fix_fork_entropy) {
  emmip_data <- emmip(rq_model, formula, at = vary_project_age_fix_fork_entropy, plotit = FALSE)
  file <- paste("./", file, sep = "/")
  write.csv(as.data.table(emmip_data),
            file <- file,
            row.names = FALSE,
            fileEncoding = "UTF-8")
  
  cat("emmip data saved to", file, "\n")
}

generate_vary_list <- function(fork_entropy_rq2_c,scale_variable_name, scale_values,fork_entropy_scale="fork_entropy_rq2_scale") {
  result_list <- list(
    fork_entropy_rq2_c,
    seq(
      scale_values[[1]],
      scale_values[[length(scale_values)]],
      by = 0.1)
  )
  names(result_list) <- c(fork_entropy_scale,scale_variable_name)
  return(result_list)
}