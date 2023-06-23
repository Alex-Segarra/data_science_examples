

run_the_sim<-function(n,months){
library(tidyverse)

fix_posix_na_and_sort<-function(x){
  x<-as.data.frame(as.POSIXct(x, '%Y-%m-%d %H:%M:%S',tz='GMT'))
  x<-x%>%
    filter(!is.na(x))
  x[,1]<-sort(x[,1]) 
  return(x)
}

add_more_timestamps<-function(x,iter){
  for(i in 1:iter){
    moretime<-rnorm(nrow(x),15,30)%>%
      round%>%
      abs()%>%
      as.integer()
      #as.integer(abs(round(rnorm(runif(nrow(x),2,360),mean=15,sd=20))))
    x2<-as.data.frame(x[,1]+moretime)
    x<-rbind(setNames(x,nm='x'),setNames(x2,nm='x'))
    }
  x[,1]<-sort(x[,1])
  timelag<-rbind(setNames(as.data.frame(0),nm='lag'),setNames(as.data.frame(diff(x[,1],1)),nm='lag'))
  x<-cbind(x,timelag)
  return(x)
}

timestamp<-''
for(i in 1:months){
  seconds<-round((runif(n,1,59)))
  minutes<-round((runif(n,1,59)))
  hours<-round(rnorm(runif(n,0,23),mean=16,sd=3))
  day<-sort(round((runif(n,1,30))))
  month<-paste0('0',i)
  ts<-paste0('2018-',month,'-',day,' ',hours,':',minutes,':',seconds)
  ts<-gsub(' ([0-9]{1}):',' 0\\1:',ts)
  ts<-gsub('(:)([0-9]{1})(:)','\\10\\2\\3',ts)
  ts<-gsub('(.*:)([0-9]{1})$','\\10\\2',ts)
  ts<-gsub('^(2018-[0-9]{2}-)([0-9]{1} )','\\10\\2',ts)
  ts<-gsub(' 24:',' 23:',ts)
  timestamp<-c(timestamp,ts)
}

timestamp<-fix_posix_na_and_sort(timestamp)

timestamp<-add_more_timestamps(timestamp,6)

head(timestamp,200)
summary(timestamp$lag)

timestamp$session<-0
i=26
for(d in 1:nrow(timestamp)){
if(timestamp$lag[d]<20){
  timestamp$session[d]<-i}
else{
  i<-i+1
  timestamp$session[d]<-i}
}

dir<-"C:/Users/r633157/Documents"
setwd(dir)
name<-read.csv('random_celeb_list.txt',header = F,stringsAsFactors = F)
name$V1<-NULL

for(i in 1:nrow(name)){
  x<-charToRaw(name$V2[i])
  name$V2[i]<-base64enc::base64encode(x)
  x<-charToRaw(name$V2[i])
  name$V3[i]<-base64enc::base64encode(x)
}

names(df)[1]<-'timestamp'

grabout<-aggregate(data=timestamp, lag ~ session, mean)

name<-name%>%
  sample_n(nrow(grabout),replace=F)
# name<-name[1:round(nrow(name)*runif(1,min=.5,max=.9)),]
#   x<-nrow(grabout)-nrow(name)
#   name2<-name[round(runif(x,1,x)),]
#   name<-rbind(name,name2)
grabout<-cbind(grabout,name)

gender<-c('M','F')
for(i in 1:nrow(grabout)){
grabout$Gender[i]<-gender[round(runif(nrow(grabout),1,2))[1]]
}
  
for(i in 1:nrow(grabout)){
  grabout$Dependents[i]<-abs(round(rnorm(runif(nrow(grabout),1,12),1.5,1)))[i]
}

group<-c('ABC Northeast', 'Trellis Telecom','Cambia','Kwest Group','Adventure Time')
for(i in 1:nrow(grabout)){
  grabout$Group[i]<-group[round(rnorm(nrow(grabout),3,.65))[1]]
}

device<-c('other','iphone','android','ipad','windows_phone')
for(i in 1:nrow(grabout)){
  grabout$Device[i]<-device[round(runif(rnorm(nrow(grabout),2,1),1,5))[1]]
}

for(i in 1:nrow(grabout)){
  grabout$Member_Age[i]<-round(runif(rnorm(nrow(grabout),35,15),18,80))[1]
}

df<-inner_join(timestamp,grabout,by=('session'))

max_sesh<-aggregate(data=df,timestamp~session,max)
min_sesh<-aggregate(data=df,timestamp~session,min)

df$Tag<-ifelse(df$timestamp%in%max_sesh$timestamp|df$session%in%max_sesh$timestamp,'sign-out',
               ifelse(df$timestamp%in%min_sesh$timestamp|df$session%in%min_sesh$timestamp,'sign-in',''))


tags<-c(
'Crash',
'Check your Care History',
'Call',
'Contact Us',
'Chat with Our Experts',
'Go to Providers',
'Item Details',
'Care Details',
'Prescriptions',
'Care Team',
'Home',
'Home',
'Home',
'Care Timeline',
'Chat with Charli',
'Provider Details',
'Go to Care Time line',
'Go to Presciptions',
'Registration Page',
'Rx Fill Details',
'Talk to Charli',
'Visit Details',
'Your Account')

for(i in 1:nrow(df)){
  df$Tag[i]<-ifelse(df$Tag[i]=='', tags[round(runif(rnorm(1,12,2),1,23))][1] ,df$Tag[i])
  df$Tag[i]<-ifelse(is.na(df$Tag[i]),'Cambia',df$Tag[i])
}

values<-c("Card : Don't Forget",'',
'Card : Help Us Learn','',
'Card : Tracking','',
'Card :Get Prepared')

df$Values<-""
for(i in 1:nrow(df)){
  if(df$Tag[i]=='Home'){
    x<-values[round(runif(1,min=1,max = length(values)))]
    df$Values[i]<-x
  }else{
    df$Values[i]<-df$Values[i]
  }
  }

df[,grep('lag',names(df))]<-NULL

names(df)[3:4]<-c('Member ID','Registration ID')
names(df)[1]<-'Timestamp'
names(df)[9]<-'Member Age'
names(df)[7]<-'Group Id'
names(df)[2]<-"Session ID"
names(df)[11]<-'Value'

return(df)
}

library(xlsx)
df<-run_the_sim(500,5)

event_report<-df[,c('Timestamp',"Session ID",'Registration ID','Tag','Value','Device')]
member_exref<-unique(df[,c('Member ID','Registration ID')])
member_demo<-unique(df[,c('Member ID','Member Age','Gender','Dependents')])

setwd('//file_path/')

'Kitchen Sink'
'Event Report'
'Member Xref'
'Member Demographics Profile'

write.csv(df,'best_sample_data_ever.csv')

write.csv(df,'Sample Data.csv', row.names = F)
write.csv(event_report,'event_report.csv', row.names = F)
write.csv(member_exref,'member_exref.csv', row.names = F)
write.csv(member_demo,'member_demo.csv', row.names = F)


