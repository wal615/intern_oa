library(tidyverse)
library(ggplot2)
library(data.table)
library(ggforce)
apply_data_raw <- read.csv("../../data/Apply_Rate_2019.csv") %>% data.table(.)
apply_data <- apply_data_raw[,!c("search_date_pacific","class_id")] 
apply_data_hist <- (apply_data) %>% gather(.) %>% 
                   ggplot(data = .) +
                   geom_histogram(aes(x =  value)) +
                   facet_wrap_paginate(facets = vars(key),
                                       scales = 'free_x',
                                       ncol = 3, 
                                       nrow = 3, 
                                       page = 1)
# adding id_freq columns
apply_data_raw[,class_id_freq := .N, by = class_id]

# change the order for following python code 
names <- colnames(apply_data_raw)[c(1:7,11,8, 9, 10)]
apply_data_raw <- apply_data_raw[,names, with = FALSE]

write.csv(apply_data_raw, file = "../../data/Apply_Rate_2019_id_freq.csv", row.names = FALSE)
