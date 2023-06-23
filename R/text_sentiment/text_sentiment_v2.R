#ALL PROPS TO Rachael Tatman for her excellent tutorial, found here:
#https://www.kaggle.com/rtatman/tutorial-sentiment-analysis-in-r
#https://arxiv.org/pdf/1712.01794.pdf

rm(list=ls())
rep(gc(),3)

library(tidyverse)
library(tidytext)
library(glue)
library(stringr)
library(gridExtra)
library(ggrepel)

dir<-"C:/Users/me/Documents"
setwd(dir)

# stick together the path to the file & 1st file name
fileName<-glue("C:/Users/me/Documents/",'feedbackAPR25.txt',sep='')

# get rid of any sneaky trailing spaces
fileName <- trimws(fileName)

# read in the new file
fileText <- glue(read_file(fileName))

# remove any dollar signs (they're special characters in R)
fileText <- gsub("\\$", "", fileText) 

#tokenize: The first thing we need to do is tokenize it, or break it into individual words.
tokens <- data_frame(text = fileText) %>% unnest_tokens(word, text)

#Create Dataset of words with values
word_deck<-rbind(get_sentiments('bing'),setNames(nma_words,nm=c('word','sentiment')),
                 c('cant','negator'),
                 c("didn't",'negator'),
                 c("couldn't",'negator'),
                 c("wouldn't",'negator'),
                 c("won't",'negator'),
                 c("didnt",'negator'),
                 c("couldnt",'negator'),
                 c("wouldnt",'negator'),
                 c("wont",'negator') )

#####################1-Gram Analysis###########
{
# get the sentiment from the first text, the TIDYR Way!: 
bing_scale<- tokens %>%
              inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
              count(sentiment) %>% # count the # of positive & negative words
              spread(sentiment, n, fill = 0) %>% # made data wide rather than narrow
              mutate(sentiment = positive - negative) %>%# # of positive words - # of negative words
              as.data.frame() %>%
              reshape2::melt()

#Attribute sentiment to it:
nrc_scale<-tokens %>%
              inner_join(get_sentiments('nrc')) %>%
              count(sentiment) %>%
              spread(sentiment, n, fill = 0) %>%
              mutate(sentiment = positive - negative) %>%
              as.data.frame()%>%
              reshape2::melt()

nrc_assign<- tokens %>%
    inner_join(get_sentiments('nrc'))%>%
  as.data.frame()

nrc_assign[nrc_assign$sentiment=='trust',]  
nrc_assign[nrc_assign$word=='doctor',] 

wordcloud<-tokens%>%
  filter(!word%in%stop_words$word)%>%
  count(word)%>%
  inner_join(get_sentiments('bing'),by='word')

wordcloud$sent_index<-if_else(wordcloud$sentiment=='positive',wordcloud$n*1,wordcloud$n*-1)

plot1<-nrc_scale[!nrc_scale$variable%in%c('negative','positive','sentiment'),] %>%
    ggplot(aes(variable,value,fill=variable))+
        geom_col()+
        coord_flip()+
        geom_label(stat='identity',nrcaes(label=value))+
        xlab('Sentiment Category')+
        ylab('Count of Sentiments')+
        ggtitle('Text Sentiment (Emotions Only)\n(National Research Council of Canada Methodology)')

plot2<-nrc_scale[nrc_scale$variable%in%c('negative','positive'),] %>%
      ggplot(aes(variable,value,fill=variable))+
        geom_col()+
        coord_flip()+
        geom_label(stat='identity',aes(label=value))+
        xlab('Sentiment')+
        ylab('Count of Sentiments')+
        ggtitle('Text Sentiments \n(National Research Council of Canada Methodology)')

plot3<-bing_scale[bing_scale$variable%in%c('negative','positive'),] %>%
       ggplot(aes(variable,value,fill=variable))+
        geom_col()+
        coord_flip()+
        geom_label(stat='identity',aes(label=value))+
        xlab('Sentiment')+
        ylab('Count of Sentiments')+
        ggtitle('(Bing Liu, Et al. methodology)')

plot4<-ggplot(wordcloud[wordcloud$n>1 & wordcloud$word!='symptoms',], aes(x=1,y=1,size=n,label=word,color=scale(sent_index)))+
  geom_text_repel(segment.size = 0, force = 1,segment.alpha=.01) +
  scale_color_gradient(low='red', high='blue',guide = F)+
  scale_size(range = c(5, 25), guide = FALSE) +
  scale_y_continuous(breaks = NULL) +
  scale_x_continuous(breaks = NULL) +
  labs(x = '', y = '') 
  
  

pdf('APR25_Text_Sentiment.pdf')
plot1
grid.arrange(plot2,plot3)
dev.off()
}

#####################SELF ADJUSTING WORD CLOUD##################
{
#n=10#####change this#
##################### 
#Use parts_of_speech, nma_words, and stop_words to filter results
add_sentimentscore<-function(dat){
  #dat<-scored_ngrams
  #CREATE VARIABLE THAT REACTS TO N 
  x<-length(grep('word',names(dat)))
  #n<-n
#LOOP TO ASSIGN SCORES  
  for(i in 1:x){
    #GRABS A COL FROM DATA
    col<-as.data.frame(dat[,i])
    #Iterating over opinion words
    col<-merge(col,word_deck,by.x=setNames(nm=names(col)), by.y='word') 
    dat<-unique(merge(dat,col,all.x=T,by.x=setNames(nm=names(dat)[i]),by.y=setNames(nm=names(col)[1])))
    #Iterating over Modifier words
    names(dat)[grep('sentiment$',names(dat))]<-paste0('sentiment',i)
}
  #SCORE OPINION WORDS
  y<-(grep('sentiment',names(dat)))
  #flip the 'Words' rows back to normal
  dat<-dat[,c(x:1,min(y):length(names(dat)))]
  #create new df that flips the order of words/  Glue it to dat
  dat<-rbind(dat,dat[,c(x:1,length(names(dat)):min(y))])
  #filter out stopwords from df
  dat<-anti_join(dat,anti_join(stop_words[,1],word_deck,by='word'),by=c("word1"="word"))
  #Create the formula
  xpr<-'0'
  sp<-paste0('dat[,',min(y),']')
  for(i in 1:x){
    colnm<-paste0('dat[,',min(y)+(i-1),']')
    xpr<-(paste0(xpr,'+',colnm,"*(1/(1.1)^",i,')'))
  }
  xpr<-(paste0(sp,'+',xpr,'-',paste0(sp,'*(1/(1.1)^1)')))
  
  #strip out adverb  and create a multiplier
  dat$adv_mult<-as.data.frame(apply(dat[,y],2, function(x)ifelse(x=='adverb',.1,0)))%>%
     lapply(function(x)ifelse(is.na(x),0,x))%>%
      as.data.frame()%>%
        rowSums()
  #strip out modals and create a multiplier
  dat$mod_mult<-as.data.frame(apply(dat[,y],2, function(x)ifelse(x=='modal',.35,0)))%>%
    lapply(function(x)ifelse(is.na(x),0,x))%>%
    as.data.frame()%>%
    rowSums()
  #strip out negator  and create a multiplier
  dat$neg_mult<-as.data.frame(apply(dat[,y],2, function(x)ifelse(x=='negator',.85,0)))%>%
    lapply(function(x)ifelse(is.na(x),0,x))%>%
    as.data.frame()%>%
    rowSums()
  
  #map values to expressions using formulas to assign values depending on whether the ngram contains
  #adverb, negators, or modals
  #FOR POSITIVES: 1*((1-dat$mod_mult+1)-1)+1*((dat$adv_mult+1)-1)+-1*((dat$neg_mult+1)-1),
  #FOR NEGATIVES: -1*((dat$adv_mult+1)-1)-1*((dat$mod_mult+1)-1)+1*((dat$neg_mult+1)-1),
  dat[,y]<-lapply(dat[,y], function(x) 
                                ifelse(x=='positive', 1+1*((dat$mod_mult+1)-1)+1*((dat$adv_mult+1)-1)-1*((dat$neg_mult+1)-1),
                                  ifelse(x=='negative',-1-1*((dat$adv_mult+1)-1)-1*((dat$mod_mult+1)-1)+1*((dat$neg_mult+1)-1),
                                         ifelse(x=='adverb'|x=='modal'|x=='negator',0,x))))
  
  #Get rid of NAs
  dat[,y]<-lapply(dat[,y],function(x)
    ifelse(is.na(x),0,x))
  #Change to numeric format
  dat[,y]<-lapply(dat[,y],as.numeric)
  #Get the Raw Sentiment Score by applying the formula built above (xpr)
  dat$raw_score<-eval(parse(text=xpr))
  #Send it!
  return(dat)
}

set_up_ngram_dataframe<-function(the_text,n){
  fix_spelling<-function(x){
    misspell<-c("cant","wont","didnt","couldnt","shouldnt","wont","dont")
    if(x%in%misspell){x<-gsub('nt',"n\\'t",x)
      }else{x<-x}
    return(x)
  } 
  #Grab File Text and create a DF
  ngrams<-data_frame(text = fileText) %>% 
    unnest_tokens(ngram, text, token='ngrams',n=1)%>%
    apply(MARGIN = c(1,2),FUN=fix_spelling)%>%
    as.data.frame()%>%
    lapply(function(x)as.character(x))%>%
    as.data.frame(stringsAsFactors=F)
  #Remove words that aren't classified as parts of speech (i.e. crap) and turn it into an n ngram df
  ngrams<-ngrams[ngrams$ngram%in%parts_of_speech$word,]%>%
    as.data.frame(stringsAsFactors=F)%>%
    collapse()%>%
    data_frame()%>%
    set_names(nm='ngram')%>%
    unnest_tokens(input='ngram',output='ngram', format='text', token='ngrams',n=n)
  #remove useless first row
  ngrams<-ngrams[-1,]
  return(ngrams)
}

make_word_clouds<-function(fileText,n){
  ngrams<-set_up_ngram_dataframe(fileText,n)
  #sets up the df to have 10 columns with names then runs it through the sentiment score function (big one above)
  scored_ngrams<-tidyr::separate(data=ngrams,col=ngram, into=c(rep(paste0('word',1:n))), sep=' ')%>%
    add_sentimentscore()
  #creates a smaller tibble thats good for ggplot to absorb
  dat_agg<-scored_ngrams%>%
    count(word1)%>%
    set_names(nm=c('word1','counter'))%>%
    inner_join(aggregate(data=scored_ngrams, raw_score ~ word1, mean))%>%
    inner_join(parts_of_speech,by=c('word1'='word'))
  #creates the wordcloud plots
  plot1<-dat_agg[grepl('Noun',dat_agg$pos,ignore.case = T),]%>%
    filter(counter>15)%>%
    filter(!word1%in%c("so",'work',"no","dont",'cool',"like",'nice','great','good','love','pretty','can','check','more','most'))%>%
    select(-starts_with("pos"))%>%
    unique()%>%
    ggplot(aes(x=1,y=1,size=counter,label=word1,color=scale(raw_score)))+
    geom_text_repel(segment.size = 0, force = 1,segment.alpha=.01) +
    scale_color_gradient2(low='red',midpoint=1, high='blue', mid='cornflowerblue', guide = 'colourbar',name='Positive(Blue)\nNegative(Red)')+
    #scale_color_gradient2(low='red',midpoint=.5, high='blue', guide = 'colourbar')+
    #scale_color_gradient(low='red',high='blue4', guide = 'colourbar')+
    scale_size(range = c(5, 25), guide = FALSE) +
    scale_y_continuous(breaks = NULL) +
    scale_x_continuous(breaks = NULL) +
    labs(x = '', y = '') +
    ggtitle('Nouns')
  
  plot2<-dat_agg[grepl('^Verb',dat_agg$pos),]%>%
    filter(counter>10)%>%
    filter(!word1%in%c("easy","doctor","page",'team','people','date','button'))%>%
    select(-starts_with("pos"))%>%
    unique()%>%
    ggplot(aes(x=1,y=1,size=counter,label=word1,color=scale(raw_score)))+
    geom_text_repel(segment.size = 0, force = 1,segment.alpha=.01) +
    #scale_color_gradient2(low='red',mid='light blue',midpoint=0, high='blue', space='lab', guide = 'colourbar')+
    scale_color_gradient2(low='red',midpoint=.5, high='blue', mid='cornflowerblue', guide = 'colourbar',name='Positive(Blue)\nNegative(Red)')+
    #scale_color_gradient(low='red',high='blue', guide = 'colourbar')+
    scale_size(range = c(5, 25), guide = FALSE) +
    scale_y_continuous(breaks = NULL) +
    scale_x_continuous(breaks = NULL) +
    labs(x = '', y = '') +
    ggtitle('Action Words')
  
  plot3<-dat_agg[grepl('adjective|adverb',dat_agg$pos,ignore.case = T),]%>%
    filter(counter>10)%>%
    filter(!word1%in%c("dental","medical"))%>%
    select(-starts_with("pos"))%>%
    unique()%>%
    ggplot(aes(x=1,y=1,size=counter,label=word1,color=scale(raw_score)))+
    geom_text_repel(segment.size = 0, force = 1,segment.alpha=.01) +
    scale_color_gradient2(low='red',midpoint=.5, high='blue', mid='cornflowerblue',name='Positive(Blue)\nNegative(Red)', guide = 'colourbar')+
    #scale_color_gradientn(colours = terrain.colors(15))+
    #guides(fill=guide_legend(title='Positive(Blue)\nNegative(Red)'))
    scale_size(range = c(5, 25), guide = FALSE) +
    scale_y_continuous(breaks = NULL) +
    scale_x_continuous(breaks = NULL) +
    labs(x = '', y = '') +
    ggtitle('Descriptives')
  
  gridExtra::grid.arrange(plot1,plot2,plot3)
  
  the_goods<-list(scored_ngrams,ngrams,dat_agg,plot1,plot2,plot3)
  
  return(the_goods)
  
}

the_goods<-make_word_clouds(fileText = fileText,7)

scored_ngrams<-as.data.frame(the_goods[1])
ngrams<-as.data.frame(the_goods[2])
dat_agg<-as.data.frame(the_goods[3])


df<-scored_ngrams%>%
  subset(raw_score>0|raw_score<0)%>%
  filter(!word1%in%c("so","no","dont","like"))
  #inner_join(scored_ngrams,dat_agg[,1:2],by='word1')%>%
  #unique()%>%
  #subset(counter>20)%>%
 

df$word1<-as.factor(df$word1)

df$dir<-as.factor(ifelse(df$raw_score>0,'positive',ifelse(df$raw_score<0,'negative','neutral')))



formula<-paste0('dir ~ ',paste0(names(df)[grep('sentiment',names(df))],collapse = ' + '))

formula2<-'dir ~ sentiment1 + sentiment2 + sentiment3 + sentiment4 + sentiment5 + sentiment6'

res<-glm(as.formula(formula), data=df, family = binomial('logit'))


# pdf('May_2_Text_Sentiment.pdf')
# gridExtra::grid.arrange(plot1,plot2,plot3)
# dev.off()

#create df that captures the two words that follow neg ngram or neg stopword
}
#df$sentiment1 + df$sentiment2 + df$sentiment3 + df$sentiment4 + df$sentiment5 + df$sentiment5 + df$sentiment6 +

