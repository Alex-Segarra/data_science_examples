rm(list=ls())
gc()

library(rvest)
library(data.table)
library(tidyverse)
library(getPass)
library(magrittr)

login_to_mpusle=function(){
  login_url='https://apps.genericappname.com/';
  
  my_session=html_session(login_url);
  login_form=html_form(my_session)[[1]];
  
  login_form$fields[[5]]$type='button';
  
  filled_login_form=set_values(login_form,`j_username`='me@company.com',`j_password`=getPass());
  submit_form(my_session,filled_login_form, submit = '<unnamed>')
  
  return(my_session)
}

#Checks to make sure that the table isn't empty
check_for_proper_stucture=function(df,txt){
  cat(paste('\nThese are the dimensions', dim(df)))
  if(dim(df)[2]>2 & dim(df)[1]>0){
    #df=df[!grepl('\\d+ - \\d+',`First Name`)][,email:=txt]
    cat('\nDing!')
    df=df[,email:=txt]
    return(df)
  }else{
    cat('\nWARNING:This website has no data. Returning Blank DF.')
    df=data.table()
    return(df)
  }
}

#Pulls the table down
grab_the_table=function(session,url,text){
  cat('\nStep 1: Navigating to ',url)
  session=jump_to(session,url);
  # print(session)
  cat('\nStep 2: Grabbing table!')
  df=read_html(session,url)%>%
    html_nodes('table')%>%
    html_table(fill=T)%>%as.data.table();
  cat('\nStep 3: Cleaning up DF and ensuring its not NULL')
  df=check_for_proper_stucture(df,text)
  cat('\nChecked for structure:', dim(df))
  return(df)
}

#Wrapper function ensures that the table is downloaded completely
capture_table=function(session,url,text,msg,cam_no,msg_no){
  #Download DF
  df=grab_the_table(session,url,text)
  #Grabs the how many pages there are
  x=df[,1][grepl('\\d+ - \\d+',get(names(df)[1]))][,1:=gsub('(.* of )(\\d+)','\\2',get(names(df)[1]))]
  #Creates a total record count and a pages count
  tot_count=as.numeric(x[,1])
  pages=ceiling(as.numeric(x[,1])/20)
  cat('\nStep 4: Making sure to capture all pages')
  cat('\nPages:', pages,'\nTotal Count: ',tot_count)
  cat('\nCampaign:',cam_no,'\nMessage: ',msg_no )
  df=df[,page:=1][,':='(camp_no=cam_no,msg_no=msg_no)]
  #recursively grabs pages based on the above variables
  grab_pages=function(df,msg,pages,tot_count,k=2,cam_no,msg_no,text){
    if(tot_count>20 & k<=pages){
      cat('\nStep 4.',k,': Grabbing Page ',k)
      sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/',msg,'?total-count=',tot_count,'&pageNo=',k,'&orderBy=&order=')
      cat('\n',sesh_url)
      df2=grab_the_table(session=my_session,url=sesh_url,text=text)
      df2=df2[,page:=k][,':='(camp_no=cam_no,msg_no=msg_no)]
      df=rbind(df,df2)
      df=grab_pages(df,msg,pages,tot_count,k=k+1,cam_no,msg_no,text)
    }else{
      df=df
    }
    return(df)
  }
  df=grab_pages(df,msg,pages,tot_count,k=2,cam_no,msg_no,text)
  cat('\nStep 5: Cleaning DF and returning')
  df=df[!grepl('\\d+ - \\d+',get(names(df)[1]))]
  return(df)
}

get_mpulse_push_dataset=function(){
  
  push_sent=data.table();
  push_not_clicked=data.table();
  push_notification_failed=data.table();
  
  #Push Campaigns and Message Nos
  nos=data.table(
    cam_numbers=c('25770','25614','25768','25613','25613','25615','25615','25615')
    ,msg_numbers=c('125125','124435','125119','124433','125121','124436','125122','125123'));
  
  for(i in 1:length(nos$cam_numbers)){
    cam_no=nos$cam_numbers[i]
    msg_no=nos$msg_numbers[i]
    
    sesh_url=paste0('https://apps.genericappname.com/reports/pn-message/',msg_no,'/performance');
    cat('\nNavigating to ',sesh_url)
    my_session=jump_to(my_session,sesh_url);
    text_to_grab=read_html(my_session,sesh_url)%>%
      html_nodes('#selected_message')%>%
      html_text();
    cat('\nGrabbing info from: ',text_to_grab)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/push-sent')
    t_push_sent=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='push-sent',cam_no,msg_no)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/push-not-clicked')
    t_push_not_clicked=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='push-not-clicked',cam_no,msg_no)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/push-failed')
    t_push_notification_failed=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='push-failed',cam_no,msg_no)
    
    push_sent = rbind(push_sent,t_push_sent)
    push_not_clicked = rbind(push_not_clicked,t_push_not_clicked)
    push_notification_failed = rbind(push_notification_failed,t_push_notification_failed)
    cat('\nNext Campaign and Message!\n')
  }
  return(list(push_sent,push_not_clicked,push_notification_failed))
}

get_mpulse_ttd_dataset=function(){
  
  sms_sent=data.table();
  sms_opt_outs=data.table();
  sms_opt_ins=data.table();
  sms_links=data.table();
  
  nos=data.table(
    cam_numbers=c('25021','25021','25021','24900','24901','24902','24185','24185')
    ,msg_numbers=c('121482','122564','122569','121011','121012','121013','115887','120884')
  );
  
  for(i in 1:length(nos$cam_numbers)){
    cam_no=nos$cam_numbers[i]
    msg_no=nos$msg_numbers[i]
    
    sesh_url=paste0('https://apps.genericappname.com/reports/sms-message/',msg_no,'/performance');
    cat('\nNavigating to ',sesh_url)
    my_session=jump_to(my_session,sesh_url);
    text_to_grab=read_html(my_session,sesh_url)%>%
      html_nodes('#selected_message')%>%
      html_text();
    cat('\nGrabbing info from: ',text_to_grab)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/sms-sent')
    t_sms_sent=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='sms-sent',cam_no,msg_no)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/sms-opt-outs')
    t_sms_opt_outs=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='sms-opt-outs',cam_no,msg_no)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/sms-opt-ins')
    t_sms_opt_ins=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='sms-opt-ins',cam_no,msg_no)
    
    sesh_url=paste0('https://apps.genericappname.com/reports/campaign/',cam_no,'/messages/',msg_no,'/sms-links')
    t_sms_links=capture_table(session=my_session,url=sesh_url,text=text_to_grab,msg='sms-links',cam_no,msg_no)
    
    sms_sent = rbind(sms_sent,t_sms_sent)
    sms_opt_outs = rbind(sms_opt_outs,t_sms_opt_outs)
    sms_opt_ins = rbind(sms_opt_ins,t_sms_opt_ins)
    sms_links = rbind(sms_links,t_sms_links)
    
    cat('\nNext Campaign and Message!\n')
  }
  return(list(sms_sent,sms_opt_outs,sms_opt_ins,sms_links))
}

my_session=login_to_mpusle()

all_df=get_mpulse_push_dataset()
push_sent=all_df[[1]]
push_not_clicked= all_df[[2]] 
push_notification_failed=all_df[[3]]

ttd_dfs=get_mpulse_ttd_dataset()
sms_sent = ttd_dfs[[1]]
sms_opt_outs = ttd_dfs[[2]]
sms_opt_ins = ttd_dfs[[3]]
sms_links = ttd_dfs[[4]]

sms_sent[,group:=gsub('([A-Za-z ]* - )(.*)','\\2',email)][,group:=gsub('v2|v1|sms 1','',group)][,group:=gsub(' -','',group)]

setwd('//company_name/Marketing/sms_and_push')

fwrite(push_sent,'push_sent.csv')
fwrite(push_not_clicked,'push_not_clicked.csv')
fwrite(push_notification_failed,'push_notification_failed.csv')
fwrite(sms_sent,'sms_sent.csv')
fwrite(sms_opt_outs,'sms_opt_outs.csv')
fwrite(sms_opt_ins,'sms_opt_ins.csv')
fwrite(sms_links,'sms_links.csv')


