library(readxl)
library(rms)
library(survival)

data <- read_excel('D:/My_Codes/lc_private_codes/R/demo_data1.xlsx', sheet = 1)

nomo <- function(data, format_str){
  data = data.frame(data)
  dd=datadist(data)
  options(datadist="dd")
  
  format_str <- parse(text = format_str)
  f1 <- eval(format_str)
  f <- lrm(f1, data = data)
  
  nom <- nomogram(f, fun= function(x)1/(1+exp(-x)),
                  lp=F, funlabel="Risk")

  plot(nom)
  
}

format_str = "label ~ FrequencySize + MaxIntensity"
nomo(data, format_str)
