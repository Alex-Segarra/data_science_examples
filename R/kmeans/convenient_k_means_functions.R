rm(list=ls())
gc()

library(data.table) #Fast Manipulation
library(tidyverse) #Because Hadley
library(cluster) #Easy Cluster Analysis
library(ggfortify) #Easy and Pretty Cluster Plots
library(foreign) # Read SPSS Files

set.seed(1)

#read file as DT


prime<-read.spss('//filepath/file.sav',to.data.frame = T)%>%
  as.data.table()

fix_dataset<-function(df){
cat('Fixing issues with data format.')
#make PID a character vector instead of factor
df$pid<-trimws(as.character(df$pid)) # Cleans PID
#kill useless columns
df[,c(names(df)[grep('pid[a-zA-Z0-9]{1}|Q9[a-zA-Z0-9]{1}$|^Q1[3-9]{1}[a-zA-Z0-9]{1}$|^Q2[0-9]{1}[a-zA-Z0-9]{1}$',ignore.case = T,names(df))]):=NULL] #removes excess columns
return(df)
}

clean_and_factor_df<-function(df){
  #df<-df[,c(1,91:101)]
  #remove NAs
  df<-df[,lapply(.SD , function(x)ifelse(is.na(x),0,x))]
  setkey(df,'pid')
  #This function creates a Binary variable for each factor in each answer
  for(i in 1:7){
    new<-df[,lapply(.SD, function(x){ifelse(x==i,1,0)}),.SDcols=2:12,by='pid']  #create 0 to 1 matrix
      setnames(new,old=names(new)[2:12], new=paste0(names(new)[2:12],'a_eq_',i)) #Dynamically Rename
      setkey(new,'pid')
      df<-df[new,nomatch=0] #append to old dataset
  }
  return(df)
}

test_clusters<-function(df){
  #Cluster Gap test
 # test<-clusGap(df[,2:length(names(df))] , FUNcluster = kmeans,K.max = 30,B=100)
  #test<-clusGap(df[,13:length(names(df))] , FUNcluster = kmeans,K.max = 30,B=100)
  ###New method from http://www.sthda.com/english/wiki/print.php?id=239
  #k_max<-30
  #wss<-sapply(1:k_max, function(k){kmeans(df[,13:length(names(df))], centers = k)$tot.withinss})
  #visualize  
  x=factoextra::fviz_nbclust(factor_df[,-c(1:12)],FUNcluster = kmeans,method = 'wss')+geom_vline(xintercept = 3)+labs(subtitle='Within Sum of Squares')
  y=factoextra::fviz_nbclust(factor_df[,-c(1:12)],FUNcluster = kmeans,method = 'gap_stat')+geom_vline(xintercept = 3)+labs(subtitle='Gap Statistic')
  z=factoextra::fviz_nbclust(factor_df[,-c(1:12)],FUNcluster = kmeans,method = 'silhouette')+geom_vline(xintercept = 3)+labs(subtitle='Silhoutte Method')
  par(mfrow=c(3,1))
  #plot(test)
  print(x)
  print(y)
  print(z)
  gridExtra::grid.arrange(x,y,z,nrow=3)
  #plot(1:k_max,wss, type='b', pch=19)
}

append_cluster_mapping<-function(x,clust){
  names(x)
  clusters<-kmeans(x[,13:length(names(x))], centers=clust, iter.max = 50, nstart=30, algorithm='Hartigan-Wong') #run kmeans
  x[,cluster:=clusters$cluster] #create new column 'cluster'
  return(x)
}

create_clusplots<-function(df2,clus){
  #holder<-df2[,2:length(names(df2))]
  pca<-princomp(df2[,])
  plot(pca$scores[,1:2])
  g<-autoplot(kmeans(x=df2,centers=clus), data=df2, frame=T, frame.type='norm')
  print(g)
}

print_all_cluster_answers<-function(df){
  for(q in 1:(dim(df)[2]-1)){
    cat('\nGraphing: ',names(df)[q])
    x<-df[,.N, by=c('Cluster',names(df)[q])]%>%
      setnames(names(df)[q],'Q')%>%
      ggplot(aes(x=Cluster,y=N,group=reorder(Q,-N),fill=Q))+
      geom_col(position = position_dodge(width = .9))+
      geom_label(aes(label=N),size=2.5, position = position_dodge(width = .9))+
      coord_flip()+
      ggtitle(names(df)[q])
    
    print(x)
  }}

print_one_cluster_cluster<-function(df,q){
    cat('\nGraphing: ',names(df)[q])
    x<-df[,.N, by=c('Cluster',names(df)[q])]%>%
      setnames(names(df)[q],'Q')%>%
      ggplot(aes(x=Cluster,y=N,group=Q,fill=Q))+
      geom_col(position = position_dodge(width = .9))+
      geom_label(aes(label=N),size=2.5, position = position_dodge(width = .9))+
      coord_flip()+
      ggtitle(names(df)[q])
    
    print(x)
  }



compute_and_append_clusters<-function(clus,test_it){
  cat('\nComputing for ',clus,' clusters!')
  dfx<-prime
  dfx<-fix_dataset(dfx)
  #Turn this on and off as needed:
  dfx<-dfx[,c(1,91:101)]
  cat('\nChanging Factor Levels to Numeric for Analysis\n')
  factor_df<-clean_and_factor_df(dfx)
  cat('\nMapping Clusters...\n')
  df_cluster<-append_cluster_mapping(factor_df[,-1],clus)
    #factor_df$cluster<-NULL
  cat('\nAppending the cluster mapping to your DF!\n')
  dfx<-cbind(dfx,df_cluster$cluster)%>%setnames('V2','Cluster')

  return(list(dfx,factor_df,clus))

}

get_it_all_and_print_pdf<-function(clus){
  #Getting All Output from function
  output<-compute_and_append_clusters(clus,test_it = T)
  df<-output[[1]]
  factor_df<-output[[2]]
  clus<-output[[3]]
  #Printing Clusplot
  cat('\nPrinting for ',clus,' clusters!')
  pdf(file=paste0('C:/Users/r633157/Documents/kmeans_for_',clus,'_clusters.pdf'),paper='a4r')  
       df[,.N,by='Cluster']%>%
        ggplot(aes(x=Cluster,y=N,color=N))+
        geom_bar(stat='identity', position = position_dodge(1))+
        geom_label(aes(label=N),position = position_dodge(1),size=3)+
        ggtitle('N per Cluster')
      create_clusplots(factor_df[,-c(1:12)],clus)
      tstcls<-test_clusters(factor_df)    
      print_all_cluster_answers(df[,-c('pid')])
     
        
      dev.off() 
 
}

do_the_k_means_thing<-function(k){
  for(i in 2:k){
    get_it_all_and_print_pdf(i)
}
}

do_the_k_means_thing(5)

#  scale_fill_gradientn(colors=rainbow(7))+
 # theme(legend.position = "")






