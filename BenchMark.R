#############################
# Titanic prediction: Bench # 
#############################

### Load packages ### 
library(pacman)
p_load(tidyverse,caret,rpart,mice,stringr,Hmisc,glmnet,gamlr,ranger,RColorBrewer,doParallel)

### Load packages ### 
vars <- c("PassengerId","Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","title",
          "famsize","mother","single","totsurvive","CabinNo")

train <- read.csv("data/train_clean.csv")%>%select(vars)%>%filter(CabinNo != "T")%>%droplevels()
test <- read.csv("data/test_clean.csv")%>%select(vars)

### Model 1: PenReg ###

# Encode some vars
train.x <- train%>%mutate_at(vars(Pclass,SibSp,Parch,famsize,totsurvive),as.factor)
test.x <- test%>%mutate_at(vars(Pclass,SibSp,Parch,famsize,totsurvive),as.factor)%>%
  select(-Survived)

train.y <- as.factor(train.x$Survived)
train.x <- model.matrix(Survived+PassengerId~-1+.,data=train.x)

test.x <- model.matrix(PassengerId~-1+.,data=test.x)

## simple covariates ##
set.seed(123)

lasso <- cv.glmnet(train.x,train.y,type.measure = "class",family = "binomial",alpha=0)
ridge <- cv.glmnet(train.x,train.y,type.measure = "class",family = "binomial",alpha=1)

lasso.pred <- predict(lasso,newx=test.x,type="class")
ridge.pred <- predict(ridge,newx=test.x,type="class")

lasso.submit <- data.frame(PassengerId=test$PassengerId, Survived = matrix(lasso.pred))
ridge.submit <- data.frame(PassengerId=test$PassengerId, Survived = matrix(ridge.pred))

table(lasso.submit$Survived)
table(ridge.submit$Survived)

write.csv(lasso.submit,file="Lasso_submit.csv",row.names = F) #0.40191
write.csv(ridge.submit,file="Ridge_submit.csv",row.names = F) #0.46411

### Model 2: PenReg with different genders ###

train.male <- filter(train,Sex == "male")%>%select(-Sex,-mother)%>%droplevels()%>%
  mutate(Survived=recode(Survived,"0"="No","1"="Yes"))
  recode(Survived,"0"="No","1"="Yes")
train.female <- filter(train,Sex == "female")%>%select(-Sex)%>%droplevels()%>%
  mutate(Survived=recode(Survived,"0"="No","1"="Yes"))
test.male <- filter(test,Sex == "male")%>%select(-Sex,-mother)%>%droplevels()
test.female <- filter(test,Sex == "female")%>%select(-Sex)%>%droplevels()

set.seed(100)

#split into train into train and csv
split.m <- createDataPartition(train.male$Survived,p=0.75,list=F)
split.f <- createDataPartition(train.female$Survived,p=0.75,list=F)

cv.train.m <- train.male[split.m,]
cv.test.m <- train.male[-split.m,]

cv.train.f <- train.female[split.f,]
cv.test.f <- train.female[-split.f,]

## model training
model.vars <- c("Pclass","Age","SibSp","Parch","Fare","Embarked","title",
                "famsize","single","totsurvive","CabinNo")
# male
m.lasso <- cv.glmnet(model.matrix(~.,data=cv.train.m[,model.vars]),cv.train.m$Survived,
                     family="binomial",alpha=0,type.measure = "class")
m.ridge <- cv.glmnet(model.matrix(~.,data=cv.train.m[,model.vars]),cv.train.m$Survived,
                     family="binomial",alpha=1,type.measure = "class")

m.lasso.pred <- predict(m.lasso,newx=model.matrix(~.,data=cv.test.m[,model.vars]),type="class")
m.ridge.pred <- predict(m.ridge,newx=model.matrix(~.,data=cv.test.m[,model.vars]),type="class")

confusionMatrix(factor(m.lasso.pred),factor(cv.test.m$Survived))
confusionMatrix(factor(m.ridge.pred),factor((cv.test.m$Survived))) # ridge is better

# female
f.lasso <- cv.glmnet(model.matrix(~.,data=cv.train.f[,model.vars]),cv.train.f$Survived,
                     family="binomial",alpha=0,type.measure = "class")
f.ridge <- cv.glmnet(model.matrix(~.,data=cv.train.f[,model.vars]),cv.train.f$Survived,
                     family="binomial",alpha=1,type.measure = "class")

f.lasso.pred <- predict(f.lasso,newx=model.matrix(~.,data=cv.test.f[,model.vars]),type="class")
f.ridge.pred <- predict(f.ridge,newx=model.matrix(~.,data=cv.test.f[,model.vars]),type="class")

confusionMatrix(factor(f.lasso.pred),factor(cv.test.f$Survived))
confusionMatrix(factor(f.ridge.pred),factor((cv.test.f$Survived))) # Ridge again does better

# Fit ridge on Male and Female separately
m.ridge <- cv.glmnet(model.matrix(~.,data=train.male[,model.vars]),train.male$Survived,
                     family="binomial",alpha=0,type.measure = "class")
f.ridge <- cv.glmnet(model.matrix(~.,data=train.female[,model.vars]),train.female$Survived,
                     family="binomial",alpha=0,type.measure = "class")

m.test.pred <- predict(m.ridge,newx=model.matrix(~.,data=test.male[,model.vars]),type="class")
f.test.pred <- predict(f.ridge,newx=model.matrix(~.,data=test.female[,model.vars]),type="class")

test.male$Survived <- m.test.pred
test.male <- test.male%>%mutate(Survived = if_else(Survived == "No",0,1))%>%select(PassengerId,Survived)

test.female$Survived <- f.test.pred
test.female <- test.female%>%mutate(Survived = if_else(Survived == "No",0,1))%>%select(PassengerId,Survived)

ridge.gender.submit <- rbind(test.male,test.female)
write.csv(ridge.gender.submit,file="Ridge_submit_gender.csv",row.names = F) #0.79904


### Model 3: RForest ###

train.male <- filter(train,Sex == "male")%>%select(-Sex,-mother)%>%droplevels()%>%
  mutate(Survived=recode(Survived,"0"="No","1"="Yes"))
train.female <- filter(train,Sex == "female")%>%select(-Sex)%>%droplevels()%>%
  mutate(Survived=recode(Survived,"0"="No","1"="Yes"))
test.male <- filter(test,Sex == "male")%>%select(-Sex,-mother)%>%droplevels()
test.female <- filter(test,Sex == "female")%>%select(-Sex)%>%droplevels()

# gender model + randomForest
rf.form <- as.formula(paste0("Survived~",paste(model.vars,collapse="+")))

test.rf <- ranger(rf.form,train.male,num.trees=500)
test.pred <- predict(test.rf,data=train.female)$predictions
test.acc <- confusionMatrix(factor(test.pred),factor(train.female$Survived))$overall["Accuracy"]

grid <- expand.grid(
  num.trees=seq(500,2000,100),mtry = seq(2,11,2),sampe_size = c(.55, .632, .70, .80),acc_m=0,acc_f=0
)

# break into cv
set.seed(100)

split.m <- createDataPartition(train.male$Survived,p=0.75,list=F)
split.f <- createDataPartition(train.female$Survived,p=0.75,list=F)

# cv.train.m <- train.male[split.m,]
# cv.test.m <- train.male[-split.m,]
# 
# cv.train.f <- train.female[split.f,]
# cv.test.f <- train.female[-split.f,]

for(i in 1:nrow(grid)){
  
  # male
  male <- ranger(rf.form,data=train.male[split.m,],
                 num.trees=grid$num.trees[i],
                 mtry=grid$mtry[i],
                 sample.fraction=grid$sampe_size[i],
                 seed=123)
  pred_male <- predict(male,data=train.male[-split.m,])$predictions
  grid$acc_m[i] <- confusionMatrix(factor(pred_male),factor(train.male[-split.m,"Survived"]))$overall["Accuracy"]
  
  # female
  female <- ranger(rf.form,data=train.female[split.f,],
                   num.trees=grid$num.trees[i],
                   mtry=grid$mtry[i],
                   sample.fraction=grid$sampe_size[i],
                   seed=123)
  pred_female <- predict(female,data=train.female[-split.f,])$predictions
  grid$acc_f[i] <- confusionMatrix(factor(pred_female),factor(train.female[-split.f,"Survived"]))$overall["Accuracy"]
}

# best male model: 500,4,0.7
# best female model: 500,6,0.632

male.rf <- ranger(rf.form,data=train.male,num.trees=500,mtry=4,sample.fraction=0.7)
female.rf <- ranger(rf.form,data=train.female,num.trees=500,mtry=6,sample.fraction=0.632)

male.predict.rf <- predict(male.rf,test.male,num.trees=500)$predictions
female.predict.rf <- predict(female.rf,test.female,num.trees=500)$predictions

test.male$Survived <- male.predict.rf 
test.female$Survived <- female.predict.rf 

submit <- bind_rows(test.male,test.female)%>%
  select(PassengerId,Survived)%>%
  mutate(Survived=if_else(Survived=="No",0,1))

write.csv(submit,file="RF_gender.csv",row.names = F) #0.79425 Not an improvement over simple Ridge; Shame

### Model 3b: K-fold RForest ###
fitControl <- trainControl(method="repeatedcv",number=4,repeats=10,classProbs = TRUE)
set.seed(1234)

male.rf.kcv <- train(rf.form,data=train.male,method="ranger",trControl=fitControl)
female.rf.kcv <- train(rf.form,data=train.female,method="ranger",trControl=fitControl)

male.predict.rf.kcv <- predict(male.rf.kcv,test.male)
female.predict.rf.kcv <- predict(female.rf.kcv,test.female)

test.male$Survived <- male.predict.rf.kcv 
test.female$Survived <- female.predict.rf.kcv 

submit.kcv <- bind_rows(test.male,test.female)%>%
  select(PassengerId,Survived)%>%
  mutate(Survived=if_else(Survived=="No",0,1))

write.csv(submit.kcv,file="RF_kcv_gender.csv",row.names = F) #0.77511

### Model 4: Surname + RandomForest/Ridge ###



### Model 5: Surname + Best ###





















