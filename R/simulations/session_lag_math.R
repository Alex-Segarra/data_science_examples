### modelin'
rm(list = ls())
library(data.table)
library(tidyverse)

set.seed(123)

graph_this = function(in_text,in_rank,in_median,in_group=''){
  #graphics.off()
  #text = gsub(' ','\\,',text) %>% 
  # strsplit(',') %>% 
  #lapply(as.numeric)
  
  text = strsplit(in_text, split = ',')%>%
    unlist%>%
    as.numeric()
  hist(text,breaks = 100,main = paste0('Rank:',in_rank,'\nVector:',unlist(in_text),'\nMedian:',in_median,'\nGroup:',in_group))
  #summary(text[[1]])
}

roll_the_dice = function(){
  
  el_sampo = data.table(stringsAsFactors = F)
  
  model_frequency = function(n){
  
      t = round(runif(n = n,min = 1,max=100))
      #cat('\nFor Sequence ',t,':')
      #cat('\nThe Summary Stats are: \n')
      #print(summary(t))
  
      session_count = n
      time_user_active = max(t)
      median_gap_sessions = median(t)
      average_gap_sessions = mean(t)
      penalty = (1 - 1/n)
      
      #(n-1)*(max(t)) * (1/median(t))
      #ans =  ((session_count * penalty) / time_user_active) * (1/median_gap_sessions) 
      ans = ((session_count -1)*time_user_active) * (1/median_gap_sessions) 
      
      #cat('The Answer is:', ans,'\n')
      
      tt = paste(unlist(t), collapse = ',')
      
      return(list(n
                  ,max(t)
                  ,min(t)
                  ,mean(t)
                  ,median(t)
                  ,ans
                  ,tt
                  ,t))
  }
  
  population_median_vector = c()
  
  for(i in 1:1000){
    n2 = round(runif(n = 1,min = 1,max=15))
    x = model_frequency(n2)
    
    sessions =unlist(x)[[1]]
    max_time = unlist(x)[[2]]
    min_time = unlist(x)[[3]]
    mean_time = unlist(x)[[4]]
    median_time = unlist(x)[[5]]
    score = unlist(x)[[6]]
    all_t = unlist(x)[[7]] 
    t = unlist(x)[[8]] 
    
    temp = data.table(sessions = sessions
                   ,max_time = max_time
                   ,min_time = min_time
                   ,mean_time = mean_time
                   ,median_time = median_time
                   ,score = score
                   ,all_t = all_t
                   ,stringsAsFactors = F)
    el_sampo = rbind(el_sampo,temp)
    
    population_median_vector = append(population_median_vector,t)%>%as.numeric
    
  }
  
  population_median_vector_ans = ((length(population_median_vector) -1 )
                                  * max(population_median_vector)
                                  * (1/median(population_median_vector)))
  
 # print("\npopulation_median_vector is: ",median(population_median_vector))
  
  el_sampo = data.table(el_sampo)
  el_sampo[,1:6 := lapply(.SD,as.numeric),.SDcols=1:6]
  el_sampo[,':='(pop_score=population_median_vector_ans
                 ,pop_med = median(population_median_vector)
                 ,rank_pure=(nrow(.SD) - rank(score)))]
  return(el_sampo)
}


df=roll_the_dice()

par(mfrow = c(3,3))

for (n in 1:9) {
  graph_this(df$all_t[n],df$rank_pure[n],df$median_time[n])
}
#df2= scale(df)%>%data.table()

quick_test = function(n,t){
  #(n-1)*(1-1/max(t)) * (1/median(t))
  (n-1)*(max(t)) * (1/median(t))
}

quick_test(4,t=c(1,2,3,4))
quick_test(4,t=c(99,100,88,78))
quick_test(10,t=c(1,2,4,66,75,88,99,90,91))
quick_test(10,t=c(1,2,3,89,99,88,99,90,91))
quick_test(10,t=c(1,2,3,4,5,6,7,8,10,44))

par(mfrow = c(1,1))

library(MASS)
boxcox(df$score+.00000000001 ~ df$sessions + df$max_time + df$median_time)
#hist(scale(sqrt(df$score)))

df[, score_normed := sqrt(score)]

df[,.(score,score_normed)]%>%melt()%>%
  ggplot()+
  geom_density(aes(x = value, group = variable, color = variable)) 


df[,':='(zscore = scale(score), zscore_normed = scale(sqrt(score)))]

df[,.(zscore,zscore_normed)]%>%melt()%>%
ggplot()+
  geom_density(aes(x = value, group = variable, color = variable)) 


df[,group:=ifelse(
            zscore_normed > 1.5, 'Frequent'
              ,ifelse(zscore_normed < -1.5, 'Infrequent'
                      ,ifelse(data.table::between(zscore_normed,0,1.5), 'Regular'
                              ,'Less than Regular')))]

df[,.N,by=group]

ggplot(data = df, aes(x=sessions,y=median_time,color=group))+
  geom_point()


library(ggfortify)

autoplot(prcomp(df[,.(sessions,median_time,max_time,min_time)],scale.	= T), data = df, colour = 'group', shape = F)

who_is_this= function(n){
  print(df[n,.(sessions, max_time, min_time, mean_time, median_time , score,zscore_normed,all_t,group)])
  graph_this(in_text = df$all_t[n], in_rank = df$rank_pure[n],in_median = df$median_time[n],in_group=df$group[n])
}

who_is_this(691)

