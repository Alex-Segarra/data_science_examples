rm(list=ls())
gc()

library(data.table)
library(tidyverse)
library(jsonlite)

dir='\\\\file_path\\'
setwd(dir)
df=data.table()
for (file in list.files(dir)) {
  temp=read.csv(file = file, header = T, sep = ",", check.names = F, stringsAsFactors = F, fill = T)%>%
    as.data.table
  names(temp)=gsub(' .*|\\..*','',names(temp))
  temp=temp[,event:=gsub('^["]+|["]+$','',event)][
    ,event:=gsub('\\\\','',event)]
  event=temp[,flatten(jsonlite::stream_in(textConnection(event)), recursive =T)]
  temp=cbind(temp,event)
  df=rbind(df,temp,fill=T)
}

df_list <- list()
for (file in list.files(dir)){  
  df = read.csv(file = file, header = T, sep = ",", check.names = F, stringsAsFactors = F, fill = T)
  names(df) = gsub(" .*",replacement = "", names(df))
  df$event = gsub('^["]+|["]+$','',df$event)
  df$event = gsub('\\\\','', df$event)
  df$event = lapply(df$event, function(j) as.list(unlist(fromJSON(j, flatten = T))))
  finaldf = df %>% mutate(event = map(event, as_tibble)) %>% unnest()
  
  df_list[[file]] <- finaldf
}

rm(finaldf)
final_df = rbindlist(df_list, fill = T) 
  
#temp=fromJSON(file, flatten = T)%>%as.data.table()
#assign(gsub('random_txt','',as.character(file)),temp)
#temp=read.csv(file)
#temp=fread(file, stringsAsFactors = F, check.names = F, header = T, sep=',', fill = T ,strip.white=T)

