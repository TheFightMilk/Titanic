#############################
# Titanic prediction: Clean # 
#############################

### Load packages ### 
library(pacman)
p_load(tidyverse,mice,stringr)

### Load Data ### 
train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

survived <- train$Survived
test$Survived <- NA
data <- rbind(train,test)

### Feature Engineering ### 

# Title
data <- data%>%rowwise()%>%
  mutate(title = str_split(Name,', ')[[1]][2],
         title = str_split(title,'\\.')[[1]][1])

mr <- c("Capt","Col","Don","Major","Rev","Jonkheer","Sir")
ms <- c("Dona","Mlle","Mme")
mrs <- c("Lady","the Countess")

data <- data%>%
  mutate(title = case_when(title=="Mr" & Age <= 14.5 & !is.na(Age) ~ "Master",
                           title %in% mr ~"Mr",
                           title %in% ms ~"Ms",
                           title %in% mrs ~ "Mrs",
                           title == "Dr" & Sex == "female" ~ "Ms",
                           title == "Dr" & Sex == "male" ~ "Mr",
                           T ~ title))

data$title <- as.factor(data$title)

# family variables
data <- data%>%
  mutate(famsize = SibSp+Parch+1,
         famsmall = if_else(famsize<=3,1,0),
         mother = if_else(title == "Mrs" & Parch > 0,1,0),
         single = if_else(famsize ==1,1,0))

# unique family identifier
data <- data%>%rowwise()%>%
  mutate(famname = str_split(Name,', ')[[1]][1])

# famticket <- filter(data,single == 0)%>%select(famname,Ticket)
data <- data%>%mutate(famname = paste0(famname,str_sub(Ticket,-3,-1)))

# how many survived in family
famsurvive <- filter(data,famsize>1)%>%
  group_by(famname)%>%
  summarise(totsurvive=sum(Survived,na.rm=T))%>%
  filter(totsurvive>0)

data <- left_join(data,famsurvive,by="famname")
data <- data%>%mutate(totsurvive=replace_na(totsurvive,0))

### Impute missing data ###
data[data$Embarked == "",]$Embarked <- NA
data[data$Cabin == "",]$Cabin <- NA

md.pattern(data)

# set embark = S
str(droplevels(data[is.na(data$Embarked),]))
data$Embarked[is.na(data$Embarked)] <-  "S"

# pred missing fare using LM
fit.fare <- lm(Fare ~ Pclass + SibSp + Parch + Age + Embarked + title, data[!is.na(data$Fare),])
data$Fare[is.na(data$Fare)] <- predict(fit.fare, newdata = data[is.na(data$Fare),])

# pred missing age using MICE
ageData <- mice(data[,!colnames(data) %in% c("Survived","Name","PassengerId","Ticket","Cabin","famname")],
                m=8,maxit=8,meth='pmm',seed=251863)
ageimp <- data.frame(data$PassengerId,complete(ageData,6))
data$Age <- ageimp$Age

data <- data%>%
  mutate(title = case_when(title == "Miss" & Age > 14.5 ~ "Ms",
                           T ~ as.character(title)))

table(data$title, data$Age > 14.5)

# cabin (copied from Kaggle website)

data$CabinNo = sapply(data$Cabin,function(x) substr(x,1,1))
data$CabinNo[data$CabinNo == ""] = NA
familyWithCabinNo = unique(data[!is.na(data$CabinNo) & data$SibSp + data$Parch > 0,c("famname", "CabinNo")])
checkIfHasCabin <- function(famname, CabinNo){   
  ifelse (famname %in% familyWithCabinNo$famname, familyWithCabinNo$CabinNo, CabinNo)      
}
data[is.na(data$CabinNo),]$CabinNo = apply(data[ is.na(data$CabinNo),c("famname", "CabinNo")], 1, function(y) checkIfHasCabin(y["famname"], y["CabinNo"]))

# for first class obs
A.1 = round(22/(323-65) * 65)
B.1 = round(65/(323-65) * 65)
C.1 = round(96/(323-65) * 65)
D.1 = round(40/(323-65) * 65)
E.1 = 65 - (A.1+B.1+C.1+D.1)
# for second class
D.2 = round(6/(277-254) * 254)
E.2 = round(4/(277-254) * 254)
F.2 = 254 - (D.2+E.2)
# for third class
E.3 = round(3/(709-691) * 691)
F.3 = round(8/(709-691) * 691)
G.3 = 691 - (E.3+F.3)

set.seed(0)
data[ sample( which( data$Pclass==1 & is.na(data$CabinNo)), A.1 ) , "CabinNo"] <- rep("A", A.1)
data[ sample( which( data$Pclass==1 & is.na(data$CabinNo)), B.1 ) , "CabinNo"] <- rep("B", B.1)
data[ sample( which( data$Pclass==1 & is.na(data$CabinNo)), C.1 ) , "CabinNo"] <- rep("C", C.1)
data[ sample( which( data$Pclass==1 & is.na(data$CabinNo)), D.1 ) , "CabinNo"] <- rep("D", D.1)
data[ sample( which( data$Pclass==1 & is.na(data$CabinNo)), E.1 ) , "CabinNo"] <- rep("E", E.1)

set.seed(0)
data[ sample( which( data$Pclass==2 & is.na(data$CabinNo)), D.2 ) , "CabinNo"] <- rep("D", D.2)
data[ sample( which( data$Pclass==2 & is.na(data$CabinNo)), E.2 ) , "CabinNo"] <- rep("E", E.2)
data[ sample( which( data$Pclass==2 & is.na(data$CabinNo)), F.2 ) , "CabinNo"] <- rep("F", F.2)

set.seed(0)
data[ sample( which( data$Pclass==3 & is.na(data$CabinNo)), E.3 ) , "CabinNo"] <- rep("E", E.3)
data[ sample( which( data$Pclass==3 & is.na(data$CabinNo)), F.3 ) , "CabinNo"] <- rep("F", F.3)
data[ sample( which( data$Pclass==3 & is.na(data$CabinNo)), G.3 ) , "CabinNo"] <- rep("G", G.3)

data$CabinNo = as.factor(data$CabinNo)
table(data$CabinNo, data$Pclass)


# Finalize
data <- select(data,-Ticket,-Name,-Cabin)

testId <- test$PassengerId
trainId <- train$PassengerId

train = data[data$PassengerId %in% trainId,]
test = data[data$PassengerId %in% testId,]

write.csv(train,"data/train_clean.csv")
write.csv(test,"data/test_clean.csv")
rm(list=ls())

















