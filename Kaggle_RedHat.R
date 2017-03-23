# https://www.kaggle.com/mauropelucchi/predicting-red-hat-business-value/explain-red-hat-data-and-xgb-model/notebook

###### Loading packages
library(zoo)
library(data.table)
library(FeatureHashing)
library(xgboost) # for extreme booster tree
library(dplyr)
library(Matrix)
library(smbinning)

###### Loading data
setwd("C:/Github/Kaggle_RedHat")

train<-fread('act_train.csv') %>% as.data.frame()
test<-fread('act_test.csv') %>% as.data.frame()
people<-fread('people.csv') %>% as.data.frame()

###### Formatting data
people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

## Convert Boolean to 1/0
people_bool <- names(people)[which(sapply(people, is.logical))]
for (col in people_bool) set(people, j = col, value = as.numeric(people[[col]]))


## Reducing group_1 and create class_1
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='default group'
people$people_class_1 <- people$people_group_1
t1 <- names(which(table(people$people_class_1) >= 350))
people$people_class_1[people$people_class_1 %in% names(which(table(people$people_class_1)< 350))]='default class'

## Reducing char_10 
unique.char_10<-
  rbind(
    select(train,people_id,char_10),
    select(test,people_id,char_10)) %>% group_by(char_10) %>% 
  summarize(n=n_distinct(people_id)) %>% 
  filter(n==1) %>% 
  select(char_10) %>%
  as.matrix() %>% 
  as.vector()

train$char_10[train$char_10 %in% unique.char_10]<-'default type'
test$char_10[test$char_10 %in% unique.char_10]<-'default type'

###### Merging datasets
d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
# set traing dataset and add outcome variable to d2
d1$train <- 1
d2$outcome <- -1
d2$train <- 0
gc()

## Handling Missing Values
d1$char_1[d1$char_1 == '']='activity type 1'
d1$char_2[d1$char_2 == '']='activity type 1'
d1$char_3[d1$char_3 == '']='activity type 1'
d1$char_4[d1$char_4 == '']='activity type 1'
d1$char_5[d1$char_5 == '']='activity type 1'
d1$char_6[d1$char_6 == '']='activity type 1'
d1$char_7[d1$char_7 == '']='activity type 1'
d1$char_8[d1$char_8 == '']='activity type 1'
d1$char_9[d1$char_9 == '']='activity type 1'
d1$char_10[d1$char_10 == '']='activity type <> 1'

d2$char_1[d2$char_1 == '']='activity type 1'
d2$char_2[d2$char_2 == '']='activity type 1'
d2$char_3[d2$char_3 == '']='activity type 1'
d2$char_4[d2$char_4 == '']='activity type 1'
d2$char_5[d2$char_5 == '']='activity type 1'
d2$char_6[d2$char_6 == '']='activity type 1'
d2$char_7[d2$char_7 == '']='activity type 1'
d2$char_8[d2$char_8 == '']='activity type 1'
d2$char_9[d2$char_9 == '']='activity type 1'
d2$char_10[d2$char_10 == '']='activity type <> 1'

result1<-smbinning(df=d1,y="outcome",x="people_char_38",p=0.1) 

gen_tmp = function(df,ivout,chrname="NewChar"){
  df=cbind(df,tmpname=NA)
  ncol=ncol(df)
  
  # Update 2016-08-20 MP
  #col_id=ivout$col_id
  col_id = which(names(df)==ivout$x) 
  
  # Updated 20160130
  b=ivout$bands
  df[,ncol][is.na(df[,col_id])]=0 # Missing
  df[,ncol][df[,col_id]<=b[2]]=1 # First valid
  # Loop goes from 2 to length(b)-2 if more than 1 cutpoint
  if (length(b)>3) {
    for (i in 2:(length(b)-2)) {
      df[,ncol][df[,col_id]>b[i] & df[,col_id]<=b[i+1]]=i
    }
  }
  df[,ncol][df[,col_id]>b[length(b)-1]]=length(b)-1 # Last
  df[,ncol]=as.factor(df[,ncol]) # Convert to factor for modeling
  blab=c(paste("01 <=",b[2]))
  if (length(b)>3) {
    for (i in 3:(length(b)-1)) {
      blab=c(blab,paste(sprintf("%02d",i-1),"<=",b[i]))
    }
  } else {i=2}
  blab=c(blab,paste(sprintf("%02d",i),">",b[length(b)-1]))
  
  # Are there ANY missing values
  # any(is.na(df[,col_id]))
  
  if (any(is.na(df[,col_id]))){
    blab=c("00 Miss",blab)
  }
  df[,ncol]=factor(df[,ncol],labels=blab)
  
  names(df)[names(df)=="tmpname"]=chrname
  return(df)
}

d1<-gen_tmp(d1,result1,"people_class_38") # Update training sample
d2<-gen_tmp(d2,result1,"people_class_38") # Update population

## Grouping char_10
d1$group_10 <- d1$char_10
d2$group_10 <- d2$char_10

group.char_10<-
  rbind(
    select(d1,people_id,group_10),
    select(d2,people_id,group_10)) %>% group_by(group_10) %>% 
  summarize(n=n_distinct(people_id)) %>% 
  filter(n > 2500) %>% 
  select(group_10) %>%
  as.matrix() %>% 
  as.vector()

d1$group_10[!(d1$group_10 %in% group.char_10)]='default group'
d2$group_10[!(d2$group_10 %in% group.char_10)]='default group'

# Features to transform - usually with large cardinality
features <- c('char_1', 'char_10', 'people_group_1')
#head(d1[, features], 20)
#head(d2[, features], 20)
# Mean of the outcome with noise
n <- nrow(d1)
for (f in features){
  g <- paste0(f,"_","p")
  #cat(f, "\n")
  outcome_mean <- tapply(train$outcome, d1[[f]], mean)
  d2[[g]] <- outcome_mean[d2[[f]]]
  d1[[g]] <- outcome_mean[d1[[f]]]
  d1[[g]] <-  rnorm(n, 1, 0.01) * (d1[[g]] * n - d1$outcome) / (n - 1) 
}

###### EDA
counts <- table(d1$outcome, d1$people_char_2)
barplot(counts, main="Red Hat - Outcome vs People Char 2",
        xlab="People Char 2", col=c("darkblue","red"),
        legend = rownames(counts))

counts <- table(d1$outcome, d1$people_char_7)
barplot(counts, main="Red Hat - Outcome vs People Char 7",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts),las=2, space = 1)

d2$day_of_week <- format(as.Date(d2$date),'%u')
d2$people_day_of_week <- format(as.Date(d2$people_date),'%u')
d1$day_of_week <- format(as.Date(d1$date),'%u')
d1$people_day_of_week <- format(as.Date(d1$people_date),'%u')


counts1 <- table(d1$outcome, d1$people_day_of_week)
barplot(counts1, main="Red Hat - Outcome vs People Day Of Week",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts1), las=2)

counts1 <- table(d1$outcome, d1$people_class_1)
counts2 <- prop.table(counts1,2)
#head(prop.table(counts1,2))
barplot(counts2, main="Red Hat - Outcome vs People class",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts2), las=2, space=1)

boxplot(d1$people_char_38~d1$outcome, 
        horizontal=T, frame=F, col="lightgray",main="Distribution") 
mtext("people_char_38~outcome",3)

smbinning.plot(result1,option="goodrate",sub="people_char_38~outcome")

counts1 <- table(d1$outcome, d1$char_1)
counts2 <- prop.table(counts1,2)
#head(prop.table(counts1,2))
barplot(counts2, main="Red Hat - Outcome vs Char 1",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts2), las=2, space=1)

counts1 <- table(d1$outcome, d1$char_2)
counts2 <- prop.table(counts1,2)
#head(prop.table(counts1,2))
barplot(counts2, main="Red Hat - Outcome vs Char 2",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts2), las=2, space=1)

counts1 <- table(d1$outcome, d1$char_5)
counts2 <- prop.table(counts1,2)
#head(prop.table(counts1,2))
barplot(counts2, main="Red Hat - Outcome vs Char 5",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts2), las=2, space=1)

counts1 <- table(d1$outcome, d1$group_10)
counts2 <- prop.table(counts1,2)
#head(prop.table(counts1,2))
barplot(counts2, main="Red Hat - Outcome vs Group 10",
        xlab=" ", col=c("darkblue","red"),
        legend = rownames(counts2), las=2, space=1)