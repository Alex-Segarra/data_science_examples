rm(list=ls())
gc()

library(data.table)
library(tidyverse)
library(forecast)

#READ IN CEP GROUPS LIST
groups<-fread('//folder/groups_list.csv')%>%
  setkey('Group Id')

mpu_full=fread('//folder/Jim/mpu_full_daily.csv')%>%
  setkey('portalid')

demog=fread('//folder/Jim/demog_data.csv')%>%
  setkey('PORTALID')

mpu_full_demog=mpu_full[demog,nomatch=0]%>%
  setkey('group_id')

mpu_full_demog=mpu_full_demog[groups,nomatch=0]

#see what the f is up
mpu_full[,daydate:=as.Date(daydate,'%Y-%m-%d')][,.N,by=daydate][,setorder(.SD,daydate)][daydate!='2018-11-08']%>%
  ggplot(aes(x=daydate,y=N))+
  geom_bar(stat='identity')+
  scale_x_date(date_breaks = ('1 weeks'))+
  theme(axis.text.x = element_text(angle = 90))

ts_dt=mpu_full_demog[Groupings=='Group',sum(vst),by=c('platform','daydate','GROUP NAME','Groupings')][
  ,setorderv(.SD,cols=c('daydate','platform'))][,daydate:=as.Date(daydate)]

ad_zeros_to_dt=function(){
  
  #Create an Array of Dates with no gaps
  dates=seq.Date(from=range(ts_dt$daydate)[1],to=range(ts_dt$daydate)[2],by='day')%>%as.data.table()
  
  #Loop through each group and platform to fill gaps
  for(k in 1:length(unique(ts_dt$platform))){
    p=unique(ts_dt$platform)[k]
    
    for(i in 1:length(unique(ts_dt$`GROUP NAME`))){
      g=unique(ts_dt$`GROUP NAME`)[i]
      cat(paste0('Checking ',p,' - ',g,' for missing dates\n'))
      
      #Get the missing Dates
      missing_dates=unique(dates[!dates[,.]%in%ts_dt[`GROUP NAME`==g & platform==p,as.Date(daydate)]])
      
      #Conditional in case the Dates are complete
      if(dim(missing_dates)[1]==0){
        cat('Looks like ',g,' has all dates!')
      }else if(dim(missing_dates)[1]>0){
        cat("Adding in Days with 0's...\n")
        add_in=data.table(platform=p,daydate=missing_dates[,1],`GROUP NAME`=g,Groupings='Group', V1=0)%>%
          setnames(2,'daydate')
        ts_dt=rbind(ts_dt,add_in)  
      }
    }
  }
  return(ts_dt)
}

ts_dt=ad_zeros_to_dt()

rmse=function(fit,actual){
  res=(fit-actual)^2
  
  x=sqrt(mean(res$V1))
  
  return(x)
}


get_estimates=function(){
  
  return_df=data.table()
  
  for(k in 1:length(unique(ts_dt$platform))){
    p=unique(ts_dt$platform)[k]
    
    for(i in 1:length(unique(ts_dt$`GROUP NAME`))){
      g=unique(ts_dt$`GROUP NAME`)[i]
      cat(paste0(p,' : ',g,'\n'))
      
      #the whole numeric vector for later...
      vector_all=ts_dt[`GROUP NAME`==g & platform==p,][
        ,setorder(.SD,daydate)][
            ,.(V1=sum(V1)),by=daydate]
      
      #Creates subsets
      create_vector_subset=function(vector,date,lag){
        vector=vector_all[data.table::between(daydate,range(ts_dt$daydate)[1],as.Date(date)-lag)][
          ,.(V1)]%>%
          ts(frequency=7)
        return(vector)
      }
      
      create_fits=function(vector,rmse=F){
        cat(paste0('Running ARIMA for the subset\n'))
        arima_subset=forecast(auto.arima(vector),h=20)
          ar_m=arima_subset$method
          arima_subset=rbind(as.data.table(as.numeric(arima_subset$fitted)),as.data.table(as.numeric(arima_subset$mean)))
        
        cat(paste0('Running Feed Forward Neural Net for the subset\n'))
        nn_subset=forecast(nnetar(y=vector,P=1,lambda='auto'),h=20)
          nn_m=nn_subset$method
          nn_subset$fitted=ifelse(is.na(nn_subset$fitted),0,nn_subset$fitted)
          nn_subset=rbind(as.data.table(as.numeric(nn_subset$fitted)),as.data.table(as.numeric(nn_subset$mean)))
          nn_subset=nn_subset[,V1:=ifelse(is.na(V1)|is.infinite(V1)|is.nan(V1),.000001,V1)][
            ,V1:=as.numeric(V1)][
              ,V1:=ifelse(V1<0,0,V1)]
          
          if(rmse==T){
            cat('Just Getting the RMSE!\n')
            vector=create_vector_subset(vector=vector_all,date='2018-07-16',lag=0)
            arima_rmse=rmse(arima_subset[,1],as.data.table(vector)[,1])
            nn_rmse=rmse(nn_subset[,1],as.data.table(vector)[,1])
            
            return(list(arima_rmse,nn_rmse))
            
          }else{
            
            return(list(arima_subset,nn_subset,ar_m,nn_m))
          }
      }
      
      #Creating Subset for training set to chose between algos
      vector_subset=create_vector_subset(vector=vector_all,date='2018-07-16',lag=20)
      
      #Getting Fitted Vectors and RMSE
      results_rmse=create_fits(vector_subset,rmse = T)
    
      arima_rmse=results_rmse[1][[1]]
      nn_rmse=results_rmse[2][[1]]

      #Creating Subset for test set
      vector_subset=create_vector_subset(vector=vector_all,date='2018-07-16',lag=0)
      
      #Getting Fitted Vectors and RMSE
      results=create_fits(vector_subset,rmse = F)
      
      arima_subset=results[1][[1]]
      nn_subset=results[2][[1]]
      ar_m=results[3][[1]]
      nn_m=results[4][[1]]
      
      zeros=rep(0,length(vector_all$V1)-length(arima_subset$V1)) %>% as.data.table() %>% setnames(1,'V1')
      
      arima_fit=rbind(arima_subset,zeros)
      
      zeros=rep(0,length(vector_all$V1)-length(nn_subset$V1)) %>% as.data.table() %>% setnames(1,'V1')
      
      nn_fit=rbind(nn_subset,zeros)
      
      #putting it all together!
      df=data.table(daydate=vector_all[,1]
                    ,nn_fit=nn_fit[,1]
                    ,arima_fit=arima_fit[,1]
                    ,original_data=vector_all[,2]
                    ,arima_method=ar_m
                    ,nn_method=nn_m
                    ,arima_rmse=arima_rmse
                    ,nn_rmse=nn_rmse)%>%
        setnames(c(1:4),c('daydate','nn_fit','arima_fit','original_data'))
      
      #Fixing pesky super high values error by taking the log of outliers
      df=df[,nn_fit:=ifelse(nn_fit>original_data^5 & original_data>1 ,log(nn_fit),nn_fit)]
      
      df=df[,':='(group=g,platform=p)]
      #gridExtra::grid.arrange(dist,forc)
      
      cat('Next...\n')
      
      return_df=rbind(return_df,df) 
      
    }}
  
  return(return_df)
}

cep_forecast=get_estimates()

#Changing NNET estimates to ARIMA Estimates in the case where it Overshoots by a shit ton
cep_forecast2=cep_forecast[,nn_fit:=ifelse(abs(as.numeric(nn_fit))>quantile(nn_fit,.99),arima_fit,nn_fit)][
  ,arima_fit:=ifelse(arima_fit<0,abs(arima_fit),arima_fit)]

pdf(file=paste0('D:/Users/me/Documents/forecast2.pdf'),paper='special',height=8.5,width = 11)
#pdf(file=paste0('c:/Users/me/Documents/forecast2.pdf'),paper='special',height=8.5,width = 11)

for(p in unique(cep_forecast2$platform)){
  for(g in unique(cep_forecast2$group)){
    cat('Graphing: ' , p , ' - ' , g , '\n')
    forc=cep_forecast[group==g & platform==p][,.(daydate,nn_fit,arima_fit,original_data)]%>%
      melt.data.table(id.vars='daydate')%>%
      ggplot(aes(x=daydate,y=value,group=variable, color=variable))+
      geom_line()+ 
      scale_x_date(date_breaks = ('1 weeks'))+
      theme(axis.text.x = element_text(angle = 90),legend.position = 'bottom')+
      ggtitle(g,subtitle = paste0(p))+
      #ggtitle(g,subtitle = paste0(p,'\n',arima_method,'\n',nn_method))+
      #labs(caption=paste0('Arima RMSE:',arima_rmse,'\nNN RMSE: ',nn_rmse))
      geom_vline(xintercept = as.numeric(as.Date(c('2018-07-16','2018-08-05'))))
    
    print(forc)
  }
}

dev.off()

#Choses the lowest RMSE to fill in vst
cep_forecast2=cep_forecast2[,vst:=ifelse(data.table::between(daydate,'2018-07-16','2018-08-05') &
                                          arima_rmse<nn_rmse,arima_fit,
                                        ifelse(data.table::between(daydate,'2018-07-16','2018-08-05') & 
                                                 arima_rmse>nn_rmse, nn_fit, original_data))]

fwrite(cep_forecast2,'//folder/Jim/mpu_daily_full_forecasted.csv')

aggregated_cep_forecast=cep_forecast[,yearmo:=paste0(month(daydate),year(daydate))]    

#####




