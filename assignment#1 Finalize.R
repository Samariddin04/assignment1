rm(list=ls())
# setwd("/Users/sofiahuang/Desktop/MSBA/machine-learning-2")

library(ISLR2)
library(tidyverse)
library(dplyr)
# install.packages('leaps')
library(leaps)
library(tree)
# install.packages('randomForest')
library(randomForest)
# install.packages('gbm')
library(gbm)
# install.packages("BART")
library(BART)
############### Problem 1 ##############
# Q1.I
attach(Hitters)
view(Hitters) # the response variable Salary is a numerical variable
dim(Hitters) # original dataset has 322 rows and 20 variables

# removing all rows with missing values
Hitters <- na.omit(Hitters) 
dim(Hitters) # after omitting the missing values we have 263 rows and 20 variables
# there are categorical variables such as league and division
# define x and y variables
x <- model.matrix(Salary~.,Hitters)[,-1] # omit intercept
y <- Hitters$Salary

# Q1.II
# best subset selection
# using maximum 6 variables
regfit.six <- regsubsets(y~x, Hitters, nvmax=6) # must define the max since the default is 8
regfit.six.sum <- summary(regfit.six)
regfit.six.sum$rsq # 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146
regfit.six.sum$rss # 36179679 30646560 29249297 27970852 27149899 26194904
regfit.six.sum$bic # -90.84637 -128.92622 -135.62693 -141.80892 -144.07143 -147.91690
regfit.six.sum$cp # 104.28132  50.72309  38.69313  27.85622  21.61301  14.02387
regfit.six.sum$adjr2 # 0.3188503 0.4208024 0.4450753 0.4672734 0.4808971 0.4972001

coef(regfit.six,6) # AtBat, Hits, Walks, CRBI, DivisionW, PutOuts

# using all variables
regfit.all <- regsubsets(y~x, Hitters, nvmax=19) 
regfit.all.sum <- summary(regfit.all)
regfit.all.sum$rsq 
# [1] 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146 0.5141227 0.5285569
# [9] 0.5346124 0.5404950 0.5426153 0.5436302 0.5444570 0.5452164 0.5454692 0.5457656
# [17] 0.5459518 0.5460945 0.5461159
regfit.all.sum$rss 
# [1] 36179679 30646560 29249297 27970852 27149899 26194904 25906548 25136930 24814051
# [10] 24500402 24387345 24333232 24289148 24248660 24235177 24219377 24209447 24201837
# [19] 24200700
regfit.all.sum$bic 
# [1]  -90.84637 -128.92622 -135.62693 -141.80892 -144.07143 -147.91690 -145.25594
# [8] -147.61525 -145.44316 -143.21651 -138.86077 -133.87283 -128.77759 -123.64420
# [15] -118.21832 -112.81768 -107.35339 -101.86391  -96.30412
regfit.all.sum$cp # looks at how much information is lost
# [1] 104.281319  50.723090  38.693127  27.856220  21.613011  14.023870  13.128474
# [8]   7.400719   6.158685   5.009317   5.874113   7.330766   8.888112  10.481576
# [15]  12.346193  14.187546  16.087831  18.011425  20.000000
regfit.all.sum$adjr2 
# [1] 0.3188503 0.4208024 0.4450753 0.4672734 0.4808971 0.4972001 0.5007849 0.5137083
# [9] 0.5180572 0.5222606 0.5225706 0.5217245 0.5206736 0.5195431 0.5178661 0.5162219
# [17] 0.5144464 0.5126097 0.5106270

# based on the RSS, all 19 variables should be chosen
reg_summary <-  summary(regfit.all)
par(mfrow = c(2,2))
plot(regfit.all.sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
rss_min = which.min(regfit.all.sum$rss) # 19
coef(regfit.all,19)
points(rss_min, regfit.all.sum$rss[rss_min], col = "red", cex = 2, pch = 20)

# based on adj r2, 11 varaibles should be chosen
plot(regfit.all.sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l") 
adj_r2_max = which.max(regfit.all.sum$adjr2) # 11
coef(regfit.all,11) # AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, LeagueN, DivisionW, PutOuts, Assists
points(adj_r2_max, regfit.all.sum$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)

# based on cp, 10 variables should be chosen
plot(regfit.all.sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
cp_min = which.min(regfit.all.sum$cp) # 10
coef(regfit.all,10) # AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, DivisionW, PutOuts, Assists
points(cp_min, regfit.all.sum$cp[cp_min], col = "red", cex = 2, pch = 20)

# based on the bic, 6 variables should be chosen
plot(regfit.all.sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
bic_min = which.min(regfit.all.sum$bic) # 6
coef(regfit.all,6) # AtBat, Hits, Walks, CRBI, DivisionW, PutOuts
points(bic_min, regfit.all.sum$bic[bic_min], col = "red", cex = 2, pch = 20)

# all of the measures of fit give different optimal number of variables because they are different ways of measuring model accuracy 
# Adj R2 is not a good measure because as you add variables, it will always go up
# MSE and RSS are also not good because the errors will keep decreasing as you add variables
# these two measures of fit, tend to overfit on training data
# BIC is the best because the number of variables that should be chosen is the lowest, 6

# Q1.III
# forward subset selection
regfit_fwd_six = regsubsets(Salary~., data = Hitters, nvmax = 6, method = "forward")
summary(regfit_fwd_six)
# based on forward selection, the best 6 variables model contains
# CRBI, Hits, PutOuts, DivisionW, ATBat, Walks
regfit_fwd_six_sum <- summary(regfit_fwd_six)
regfit_fwd_six_sum$rsq # 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146
regfit_fwd_six_sum$rss # 36179679 30646560 29249297 27970852 27149899 26194904
regfit_fwd_six_sum$bic # -90.84637 -128.92622 -135.62693 -141.80892 -144.07143 -147.91690
regfit_fwd_six_sum$cp # 104.28132  50.72309  38.69313  27.85622  21.61301  14.02387
regfit_fwd_six_sum$adjr2 # 0.3188503 0.4208024 0.4450753 0.4672734 0.4808971 0.4972001

regfit_fwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "forward")
regfit_fwd_all_sum <- summary(regfit_fwd)
regfit_fwd_all_sum$rsq
# [1] 0.3214501 0.4252237 0.4514294 0.4754067 0.4908036 0.5087146 0.5132286 0.5281386
# [9] 0.5346124 0.5404950 0.5426153 0.5436302 0.5444570 0.5452164 0.5454692 0.5457656
# [17] 0.5459518 0.5460945 0.5461159
regfit_fwd_all_sum$rss
# [1] 36179679 30646560 29249297 27970852 27149899 26194904 25954217 25159234 24814051
# [10] 24500402 24387345 24333232 24289148 24248660 24235177 24219377 24209447 24201837
# [19] 24200700
regfit_fwd_all_sum$bic 
# [1]  -90.84637 -128.92622 -135.62693 -141.80892 -144.07143 -147.91690 -144.77245
# [8] -147.38199 -145.44316 -143.21651 -138.86077 -133.87283 -128.77759 -123.64420
# [15] -118.21832 -112.81768 -107.35339 -101.86391  -96.30412
regfit_fwd_all_sum$cp 
# [1] 104.281319  50.723090  38.693127  27.856220  21.613011  14.023870  13.607126
# [8]   7.624674   6.158685   5.009317   5.874113   7.330766   8.888112  10.481576
# [15]  12.346193  14.187546  16.087831  18.011425  20.000000
regfit_fwd_all_sum$adjr2 
# [1] 0.3188503 0.4208024 0.4450753 0.4672734 0.4808971 0.4972001 0.4998663 0.5132768
# [9] 0.5180572 0.5222606 0.5225706 0.5217245 0.5206736 0.5195431 0.5178661 0.5162219
# [17] 0.5144464 0.5126097 0.5106270

# based on rss, all 19 variables should be chosen
which.min(regfit_fwd_all_sum$rss) # 19
coef(regfit_fwd,19) 

# based on adj r2, 11 variables should be chosen 
# AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, LeagueN, DivisionW, PutOuts, Assists
which.max(regfit_fwd_all_sum$adjr2) # 11
coef(regfit_fwd,11) 

# based on bic, 6 variables should be chosen
# AtBat, Hits, Walks, CRBI, DivisionW, PutOuts
which.min(regfit_fwd_all_sum$bic) # 6
coef(regfit_fwd,6) 

# based on cp, 10 variables should be chosen
# AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, DivisionW, PutOuts, Assists
which.min(regfit_fwd_all_sum$cp) # 10
coef(regfit_fwd,10) 


# Q1.IV
# backward subset selection
regfit_bwd_six = regsubsets(Salary~., data = Hitters, nvmax = 6, method = "backward")
summary(regfit_bwd_six)
# based on backward selection, the best 6 variables model contains
# CRuns, Hits, PutOuts, AtBat, Walks, DivisionW
regfit_bwd_six_sum <- summary(regfit_bwd_six)
regfit_bwd_six_sum$rsq # 0.3166062 0.4147791 0.4484661 0.4664051 0.4840589 0.4997274
regfit_bwd_six_sum$rss # 36437951 31203460 29407297 28450807 27509524 26674092
regfit_bwd_six_sum$bic # -88.97559 -124.18997 -134.21006 -137.33435 -140.61064 -143.14927
regfit_bwd_six_sum$cp # 106.87463  56.31494  40.27961  32.67547  25.22401  18.83541
regfit_bwd_six_sum$adjr2 # 0.3139878 0.4102774 0.4420777 0.4581323 0.4740211 0.4880022

regfit_bwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "backward")
regfit_bwd_all_sum <- summary(regfit_bwd)
regfit_bwd_all_sum$rsq
# [1] 0.3166062 0.4147791 0.4484661 0.4664051 0.4840589 0.4997274 0.5136174 0.5281386
# [9] 0.5346124 0.5404950 0.5426153 0.5436302 0.5444570 0.5452164 0.5454692 0.5457656
# [17] 0.5459518 0.5460945 0.5461159
regfit_bwd_all_sum$rss
# [1] 36437951 31203460 29407297 28450807 27509524 26674092 25933487 25159234 24814051
# [10] 24500402 24387345 24333232 24289148 24248660 24235177 24219377 24209447 24201837
# [19] 24200700
regfit_bwd_all_sum$bic 
# [1]  -88.97559 -124.18997 -134.21006 -137.33435 -140.61064 -143.14927 -144.98259
# [8] -147.38199 -145.44316 -143.21651 -138.86077 -133.87283 -128.77759 -123.64420
# [15] -118.21832 -112.81768 -107.35339 -101.86391  -96.30412
regfit_bwd_all_sum$cp 
# [1] 106.874632  56.314938  40.279613  32.675465  25.224013  18.835412  13.398979
# [8]   7.624674   6.158685   5.009317   5.874113   7.330766   8.888112  10.481576
# [15]  12.346193  14.187546  16.087831  18.011425  20.000000
regfit_bwd_all_sum$adjr2 
# [1] 0.3139878 0.4102774 0.4420777 0.4581323 0.4740211 0.4880022 0.5002657 0.5132768
# [9] 0.5180572 0.5222606 0.5225706 0.5217245 0.5206736 0.5195431 0.5178661 0.5162219
# [17] 0.5144464 0.5126097 0.5106270

# based on rss, all 19 variables should be chosen
which.min(regfit_bwd_all_sum$rss) # 19
coef(regfit_bwd,19) 

# based on adj r2, 11 variables should be chosen 
# AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, LeagueN, DivisionW, PutOuts, Assists
which.max(regfit_bwd_all_sum$adjr2) # 11
coef(regfit_bwd,11) 

# based on bic, 6 variables should be chosen
# AtBat, Hits, Walks, CRuns, CRBI, CWalks, DivisionW, PutOuts
which.min(regfit_bwd_all_sum$bic) # 8
coef(regfit_bwd,8) 

# based on cp, 10 variables should be chosen
# AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, DivisionW, PutOuts, Assists
which.min(regfit_bwd_all_sum$cp) # 10
coef(regfit_bwd,10) 

# Q1.V
# best subset model
#Looking at R2
#p=6
#subset model
regfit.six.sum$rsq
#0.5087146 
#Forward
regfit_fwd_six_sum$rsq 
#0.5087146
#Backwards
regfit_bwd_six_sum$rsq
#0.4997274 

# Looking at Adjr2
#p=6
#subset model
regfit.six.sum$adjr2
# 0.4972001
#Forward
regfit_fwd_six_sum$adjr2 
# 0.4972001
#Backwards
regfit_bwd_six_sum$adjr2
#0.4880022

# Looking at RSS
#p=6
#subset model
regfit.six.sum$rss
# 26194904
#Forward
regfit_fwd_six_sum$rss
# 26194904
#Backwards
regfit_bwd_six_sum$rss
#26674092

#BIC 
#subset model
regfit.six.sum$bic
# -147.91690
#Forward
regfit_fwd_six_sum$bic
#-147.91690
#Backwards
regfit_bwd_six_sum$bic
#-143.14927

# Looking at cp
#p=6
#subset model
regfit.six.sum$cp
# 14.02387
#Forward
regfit_fwd_six_sum$cp
# 14.02387
#Backwards
regfit_bwd_six_sum$cp
#18.83541




#p=19
#subset model
#rsq
regfit.all.sum$rsq
#0.5461159

#Forward
regfit_fwd_all_sum$rsq
# 0.5461159

#Backwards
regfit_bwd_all_sum$rsq
# 0.5461159

#AdjR2
regfit.all.sum$adjr2
# 0.5106270

#Forward
regfit_fwd_all_sum$adjr2
# 0.5106270

#Backwards
regfit_bwd_all_sum$adjr2
# 0.5106270

#rss
regfit.all.sum$rss
# 24200700

#Forward
regfit_fwd_all_sum$rss
# 24200700

#Backwards
regfit_bwd_all_sum$rss
# 24200700

#BIC
regfit.all.sum$bic
# -96.30412

#Forward
regfit_fwd_all_sum$bic
# -96.30412

#Backwards
regfit_bwd_all_sum$bic
# -96.30412

#CP ##ASK ##
regfit.all.sum$cp
# 20.000000

#Forward
regfit_fwd_all_sum$cp
# 20.000000

#Backwards
regfit_bwd_all_sum$cp
# 20.000000
# best forward selection model

# best backward selection model

############ Problem 2 #############
# Q2.I
attach(Carseats)
view(Carseats) # the response variable Sales is a numerical variable
dim(Carseats) # original dataset has 400 rows and 11 variables

# removing all rows with missing values
Carseats <- na.omit(Carseats) 
dim(Carseats) # there were no missing values to omit
# there are categorical variables such as ShelveLoc and Urban
# define x and y variables
x <- model.matrix(Sales~.,Carseats)[,-1] # omit intercept
y <- Carseats$Sales

# Q2.II
# best subset selection
# using all variables
regfit_all <- regsubsets(y~x, Carseats, nvmax=10) 
regfit_all_sum <- summary(regfit_all)
regfit_all_sum$rsq
# [1] 0.2505104 0.4699029 0.6299770 0.7124094 0.7782355 0.8478863 0.8719825 0.8724943
# [9] 0.8729481 0.8733106
regfit_all_sum$rss 
# [1] 2385.0818 1686.9145 1177.5148  915.1924  705.7155  484.0675  407.3869  405.7583
# [9]  404.3142  403.1604
regfit_all_sum$bic 
# [1] -103.3622 -235.9037 -373.7102 -468.5296 -566.5070 -711.3106 -774.3036 -769.9144
# [9] -765.3491 -760.5007
regfit_all_sum$cp 
# [1] 1901.256110 1230.797363  742.155197  491.492304  291.728975   80.242677
# [7]    8.385679    8.817079    9.426118   10.314876
regfit_all_sum$adjr2 
# [1] 0.2486272 0.4672324 0.6271738 0.7094971 0.7754212 0.8455640 0.8696965 0.8698854
# [9] 0.8700161 0.8700538

# based on the RSS, all 10 variables should be chosen
par(mfrow = c(2,2))
plot(regfit_all_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
which.min(regfit_all_sum$rss) # 10
coef(regfit_all,10) 
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes
points(10, regfit_all_sum$rss[10], col = "red", cex = 2, pch = 20)

# based on adj r2, 10 varaibles should be chosen
plot(regfit_all_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l") 
which.max(regfit_all_sum$adjr2) # 10
coef(regfit_all,10) 
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes
points(10, regfit_all_sum$adjr2[10], col ="red", cex = 2, pch = 20)

# based on cp, 7 variables should be chosen
plot(regfit_all_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(regfit_all_sum$cp) # 7
coef(regfit_all,7) 
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age
points(7, regfit_all_sum$cp[7], col = "red", cex = 2, pch = 20)

# based on the bic, 7 variables should be chosen
plot(regfit_all_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
which.min(regfit_all_sum$bic) # 7
coef(regfit_all,7) 
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age
points(7, regfit_all_sum$bic[7], col = "red", cex = 2, pch = 20)

# RSS and AdjR2 both give the same answer of 10 variables and CP and BIC both say 7 variables should be chosen
# there is no correct best answer 

# Q2.III
# forward selection
regfit_fwd = regsubsets(Sales~., data = Carseats, nvmax = 10, method = "forward")
regfit_fwd_all_sum <- summary(regfit_fwd)
regfit_fwd_all_sum$rsq
# [1] 0.2505104 0.4699029 0.6299770 0.7124094 0.7782355 0.8478863 0.8719825 0.8724943
# [9] 0.8729481 0.8733106
regfit_fwd_all_sum$rss
# [1] 2385.0818 1686.9145 1177.5148  915.1924  705.7155  484.0675  407.3869  405.7583
# [9]  404.3142  403.1604
regfit_fwd_all_sum$bic 
# [1] -103.3622 -235.9037 -373.7102 -468.5296 -566.5070 -711.3106 -774.3036 -769.9144
# [9] -765.3491 -760.5007
regfit_fwd_all_sum$cp 
# [1] 1901.256110 1230.797363  742.155197  491.492304  291.728975   80.242677
# [7]    8.385679    8.817079    9.426118   10.314876
regfit_fwd_all_sum$adjr2 
# [1] 0.2486272 0.4672324 0.6271738 0.7094971 0.7754212 0.8455640 0.8696965 0.8698854
# [9] 0.8700161 0.8700538

par(mfrow = c(2,2))
# based on rss, all 10 variables should be chosen
plot(regfit_fwd_all_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
which.min(regfit_fwd_all_sum$rss) # 10
coef(regfit_fwd,10) 
points(10, regfit_fwd_all_sum$rss[10], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes

# based on adj r2, 11 variables should be chosen 
plot(regfit_fwd_all_sum$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l") 
which.max(regfit_fwd_all_sum$adjr2) # 10
coef(regfit_fwd,10) 
points(10, regfit_fwd_all_sum$adjr2[10], col ="red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes

# based on bic, 7 variables should be chosen
plot(regfit_fwd_all_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
which.min(regfit_fwd_all_sum$bic) # 7
coef(regfit_fwd,7) 
points(7, regfit_fwd_all_sum$bic[7], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age

# based on cp, 7 variables should be chosen
plot(regfit_fwd_all_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(regfit_fwd_all_sum$cp) # 7
coef(regfit_fwd,7) 
points(7, regfit_fwd_all_sum$cp[7], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age



# Q2.IV
# backward selection
regfit_bwd = regsubsets(Sales~., data = Carseats, nvmax = 10, method = "backward")
regfit_bwd_all_sum <- summary(regfit_bwd)
regfit_bwd_all_sum$rsq
# [1] 0.2505104 0.4699029 0.6299770 0.7124094 0.7782355 0.8478863 0.8719825 0.8724943
# [9] 0.8729481 0.8733106
regfit_bwd_all_sum$rss
# [1] 2385.0818 1686.9145 1177.5148  915.1924  705.7155  484.0675  407.3869  405.7583
# [9]  404.3142  403.1604
regfit_bwd_all_sum$bic 
# [1] -103.3622 -235.9037 -373.7102 -468.5296 -566.5070 -711.3106 -774.3036 -769.9144
# [9] -765.3491 -760.5007
regfit_bwd_all_sum$cp 
# [1] 1901.256110 1230.797363  742.155197  491.492304  291.728975   80.242677
# [7]    8.385679    8.817079    9.426118   10.314876
regfit_bwd_all_sum$adjr2 
# [1] 0.2486272 0.4672324 0.6271738 0.7094971 0.7754212 0.8455640 0.8696965 0.8698854
# [9] 0.8700161 0.8700538

# based on rss, all 10 variables should be chosen
plot(regfit_bwd_all_sum$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
which.min(regfit_bwd_all_sum$rss) # 10
coef(regfit_bwd,10) 
points(10, regfit_bwd_all_sum$rss[10], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes

# based on adj r2, 10 variables should be chosen 
plot(regfit_bwd_all_sum$adjr2, xlab = "Number of Variables", ylab = "Adj Rsq", type = "l")
which.max(regfit_bwd_all_sum$adjr2) # 10
coef(regfit_bwd,10) 
points(10, regfit_bwd_all_sum$adjr2[10], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age, Education, UrbanYes, USYes

# based on bic, 7 variables should be chosen
plot(regfit_bwd_all_sum$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
which.min(regfit_bwd_all_sum$bic) # 7
coef(regfit_bwd,7) 
points(7, regfit_bwd_all_sum$bic[7], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age

# based on cp, 7 variables should be chosen
plot(regfit_bwd_all_sum$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(regfit_bwd_all_sum$cp) # 7
coef(regfit_bwd,7) 
points(7, regfit_bwd_all_sum$cp[7], col = "red", cex = 2, pch = 20)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age

# Q2.V
# final model decision
# i would include 7 variables in my final model
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age
# these 7 were included by all of the accuracy measures we looked at while Education, UrbanYes, and USYes were only included when we measured with RSS and AdjR2

# Q2.VI
# extra credit - validation set 
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Carseats),
                  replace = TRUE)
test <- (!train)
regfit_best <- regsubsets(Sales~., data = Carseats[train, ], nvmax = 10)
test_matrix <- model.matrix(Sales~., data = Carseats[test, ])

val_errors <- rep(NA, 10)
for (i in 1:10) {
  coefs <- coef(regfit_best, id = i)
  pred <- test_matrix[, names(coefs)] %*% coefs
  val_errors[i] <- mean((Carseats$Sales[test] - pred)^2)
}
val_errors
# [1] 5.937068 4.132898 2.832809 2.254129 1.802738 1.248974 1.075752 1.079303 1.083295
# [10] 1.080387
which.min(val_errors) # 7
coef(regfit_best, 7)
# CompPrice, Income, Advertising, Price, ShelveLocGood, ShelveLocMedium, Age

# extra credit - cross validation 
predict.regsubsets <- function(object, newdata , id) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
  }

k <- 10
n <- nrow(Carseats)
set.seed(1)
folds <- sample(rep(1:k, length = n))
cv_errors <- matrix(NA, k, 10,
                      dimnames = list(NULL, paste(1:10)))
for (j in 1:k) {
  best_fit <- regsubsets(Sales~.,data = Carseats[folds != j, ],nvmax = 10)
  for (i in 1:10) {
    pred <- predict(best_fit, Carseats[folds == j, ], id = i)
    cv_errors[j, i] <-
      mean((Carseats$Sales[folds == j] - pred)^2)
    }
  }
apply(cv_errors, 2, mean)
# 1        2        3        4        5        6        7        8        9       10 
# 6.033689 4.284327 3.017323 2.375481 1.973746 1.259332 1.071309 1.088966 1.092923 1.087530 
par(mfrow = c(1, 1))
plot(apply(cv_errors, 2, mean), type = "b")
# 7 variable model is the best
reg_best <- regsubsets(Sales~., data = Carseats,nvmax = 10)
coef(reg_best, 10)
# (Intercept)       CompPrice          Income     Advertising           Price 
# 5.76189929      0.09260934      0.01577395      0.12504441     -0.09530210 
# ShelveLocGood ShelveLocMedium             Age       Education        UrbanYes 
# 4.84673625      1.95214520     -0.04611922     -0.02241123      0.11885295 
# USYes 
# -0.19907477 

############### Problem 3 #################
# shrinkage methods
# Q3.I
# install.packages('glmnet')
library(glmnet)

x <- model.matrix(Salary~., Hitters)[, -1]
y <- Hitters$Salary
grid <- 10^seq(10, -2, length = 100)
ridge_model <- glmnet(x, y, alpha = 0, lambda = grid) # alpha = 0 is ridge; alpha = 1 is lasso

# printing the lambda grid and the corresponding l2 norm
ridge_model$lambda
l2_norms <- sqrt(colSums(coef(ridge_model)[-1, ]^2))
l2_norms

sqrt(sum(coef(ridge_model)[-1,50]^2))
# using predict() to obtain the ridge regression coefficients
predict(ridge_model, s = 100, type = "coefficients")[1:20, ]

# Q3.II
set.seed(1)
train_ridge <- sample(1:nrow(x), nrow(x) / 2)
test_ridge <- (-train_ridge)
y_test_ridge <- y[test_ridge]
# fitting a ridge regression on the training set
ridge_model <- glmnet(x[train_ridge, ], y[train_ridge], alpha = 0,
                    lambda = grid, thresh = 1e-12)
# evaluating MSE on the test set
ridge_pred <- predict(ridge_model, s = 4, newx = x[test_ridge, ])
mean((ridge_pred - y_test_ridge)^2)
# 142199.2

# Q3.III
# using cross-validation to find the best tuning parameter lambda and corresponding lambda
set.seed(1)
cv_out <- cv.glmnet(x[train_ridge, ], y[train_ridge], alpha = 0)
plot(cv_out)
best_lambda <- cv_out$lambda.min
best_lambda # 326.0828
ridge_pred <- predict(ridge_model, s = best_lambda,
                      newx = x[test_ridge, ])
mean((ridge_pred - y_test_ridge)^2)
# 139856.6
out <- glmnet(x, y, alpha = 0, lambda = grid)
# coefficients 
predict(out, type = "coefficients",
        s = best_lambda)[1:20, ]

# Q3.IV
# lasso regression
grid <- 10^seq(10, -2, length = 100)
lasso_model <- glmnet(x, y, alpha = 1, lambda = grid)
predict(lasso_model, s = 100, type = "coefficients")[1:20, ]
options(scipen=999)
set.seed(1)
train_lasso <- sample(1:nrow(x), nrow(x) / 2)
test_lasso <- (-train_lasso)
y_test_lasso <- y[test_lasso]
# fitting a lasso regression on the training set
lasso_model <- glmnet(x[train_lasso, ], y[train_lasso], alpha = 1,
                      lambda = grid, thresh = 1e-12)
# evaluating MSE on the test set
lasso_pred <- predict(lasso_model, s = 4, newx = x[test_lasso, ])
mean((lasso_pred - y_test_lasso)^2) 
# 145298

set.seed(1)
cv_out <- cv.glmnet(x[train_lasso, ], y[train_lasso], alpha = 1)
plot(cv_out)
best_lambda <- cv_out$lambda.min
best_lambda # 9.286955
lasso_pred <- predict(lasso_model, s = best_lambda,
                      newx = x[test_lasso, ])
mean((lasso_pred - y_test_lasso)^2)
# 143572.1
out <- glmnet(x, y, alpha = 1, lambda = grid)
# coefficients 
predict(out, type = "coefficients",
        s = best_lambda)[1:20, ]

############### Problem 4 #################
# shrinkage methods
# Q4.I
x <- model.matrix(Sales~., Carseats)[, -1]
y <- Carseats$Sales

grid <- 10^seq(10, -2, length = 100)
ridge_model <- glmnet(x, y, alpha = 0, lambda = grid) # alpha = 0 is ridge; alpha = 1 is lasso

# printing the lambda grid and the corresponding l2 norm
ridge_model$lambda
l2_norms <- sqrt(colSums(coef(ridge_model)[-1, ]^2))
l2_norms

sqrt(sum(coef(ridge_model)[-1,50]^2))
# sing predict() to obtain the ridge regression coefficients
predict(ridge_model, s = 100, type = "coefficients")[1:12, ]

# Q4.II
set.seed(1)
train_ridge <- sample(1:nrow(x), nrow(x) / 2)
test_ridge <- (-train_ridge)
y_test_ridge <- y[test_ridge]
# fitting a ridge regression on the training set
ridge_model <- glmnet(x[train_ridge, ], y[train_ridge], alpha = 0,
                      lambda = grid, thresh = 1e-12)
# evaluating MSE on the test set
ridge_pred <- predict(ridge_model, s = 500, newx = x[test_ridge, ])
mean((ridge_pred - y_test_ridge)^2)
# 8.006464

# Q4.III
# using cross-validation to find the best tuning parameter lambda and corresponding lambda
set.seed(1)
cv_out <- cv.glmnet(x[train_ridge, ], y[train_ridge], alpha = 0)
plot(cv_out)
best_lambda <- cv_out$lambda.min
best_lambda # 0.1299233
ridge_pred <- predict(ridge_model, s = best_lambda,
                      newx = x[test_ridge, ])
mean((ridge_pred - y_test_ridge)^2)
# 1.020898
out <- glmnet(x, y, alpha = 0, lambda = grid)
# coefficients 
predict(out, type = "coefficients",
                        s = best_lambda)[1:12, ]

# Q4.IV
# lasso regression
grid <- 10^seq(10, -2, length = 100)
lasso_model <- glmnet(x, y, alpha = 1, lambda = grid)
predict(lasso_model, s = 100, type = "coefficients")[1:12, ]
options(scipen=999)

set.seed(1)
train_lasso <- sample(1:nrow(x), nrow(x) / 2)
test_lasso <- (-train_lasso)
y_test_lasso <- y[test_lasso]
# fitting a lasso regression on the training set
lasso_model <- glmnet(x[train_lasso, ], y[train_lasso], alpha = 1,
                      lambda = grid, thresh = 1e-12)
# evaluating MSE on the test set
lasso_pred <- predict(lasso_model, s = 500, newx = x[test_lasso, ])
mean((lasso_pred - y_test_lasso)^2) 
# 8.061331
set.seed(1)
cv_out <- cv.glmnet(x[train_lasso, ], y[train_lasso], alpha = 1)
plot(cv_out)
best_lambda <- cv_out$lambda.min
best_lambda # 0.009381517
lasso_pred <- predict(lasso_model, s = best_lambda,
                      newx = x[test_lasso, ])
mean((lasso_pred - y_test_lasso)^2)
# 0.9701046
out <- glmnet(x, y, alpha = 1, lambda = grid)
# coefficients 
predict(out, type = "coefficients",
        s = best_lambda)[1:12, ]

################## Problem 5 #####################
# fit a classification tree
rm(list=ls())
# install.packages('tree')
library(tree)
library(ISLR2)
attach(Carseats)
# Q5.I
# recoding numerical response variable to binary categorical
High <- as.factor(ifelse(Sales <= 8, "No", "Yes"))
Carseats <- data.frame(Carseats, High)
# Carseats$High <- as.factor(High)

# Q5.II
# fitting tree
tree_carseats <- tree(High ~ . - Sales, Carseats)
summary(tree_carseats)

# Q5.III
par(mfrow = c(1,1))
plot(tree_carseats)
text(tree_carseats, pretty = 0)

# Q5.IV
# printing output corresponding to each branch
tree_carseats

# Q5.V
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
test <- Carseats[-train, ]
High_test <- High[-train]
tree_train <- tree(High ~ . - Sales, Carseats,
                        subset = train)
plot(tree_train)
text(tree_train, pretty = 0)
summary(tree_train)

# Q5.VI
tree_pred <- predict(tree_train, test,
                       type = "class")
table(tree_pred, High_test)
# test error rate
(13+33)/200 # = 0.23

# Q5.VII
# pruning tree
set.seed(7)
cv_carseats <- cv.tree(tree_train, FUN = prune.misclass)
cv_carseats$dev # 9 terminal nodes has †he lowest error of 74
cv_carseats$size
cv_carseats$k # best k is 1.4
par(mfrow = c(1, 2))
plot(cv_carseats$size, cv_carseats$dev, type = "b")
plot(cv_carseats$k, cv_carseats$dev, type = "b")

# Q5.VIII
prune_carseats <- prune.misclass(tree_train, best = 9)
plot(prune_carseats)
text(prune_carseats, pretty = 0)

tree_pred <- predict(prune_carseats, test, type = "class")
summary(prune_carseats)
table(tree_pred, High_test) 
# error rate: 0.105
(7+14)/200
High_test
################## Problem 6 ######################
rm(list=ls())
attach(Boston)

# Q6.I
Price <- as.factor(ifelse(medv <= 15, "low", ifelse(medv > 15 & medv <= 30, "mid", "high")))
Boston <- data.frame(Boston, Price)

# Q6.II
tree_boston <- tree(Price ~ . - medv, Boston)
summary(tree_boston)

# Q6.III
par(mfrow = c(1,1))
plot(tree_boston)
text(tree_boston, pretty = 0)

# Q6.IV
# printing output corresponding to each branch
tree_boston

# Q6.V
set.seed(5)
train <- sample(1:nrow(Boston), 300)
test <- Boston[-train, ]
Price_test <- Price[-train]
tree_train <- tree(Price ~ . - medv, Boston,
                   subset = train)
plot(tree_train)
text(tree_train, pretty = 0)
summary(tree_train)

# Q6.VI
tree_pred <- predict(tree_train, test,
                     type = "class")
table(tree_pred, Price_test)
# test error rate
(6 + 12 + 8 + 9)/206 # 0.1699029

# Q6.VII
# prune the tree
set.seed(7)
cv_boston <- cv.tree(tree_train, FUN = prune.misclass)
cv_boston$dev # 14 terminal nodes has †he lowest error of 34
cv_boston$size
cv_boston$k # best k is -inf
par(mfrow = c(1, 2))
plot(cv_boston$size, cv_boston$dev, type = "b")
plot(cv_boston$k, cv_boston$dev, type = "b")

# Q6.VIII
prune_boston <- prune.misclass(tree_train, best = 14)
plot(prune_boston)
text(prune_boston, pretty = 0)

tree_pred <- predict(prune_boston, test, type = "class")
summary(prune_boston)
table(tree_pred, Price_test) 

################# Problem 7 ###############
# regression trees
rm(list=ls())
attach(Boston)

# Q7.I
tree_boston <- tree(medv ~ ., Boston)
summary(tree_boston)
par(mfrow = c(1,1))
plot(tree_boston)
text(tree_boston, pretty = 0)

# Q7.II
# training data
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
tree_boston <- tree(medv ~ ., Boston, subset = train)
summary(tree_boston)
plot(tree_boston)
text(tree_boston, pretty = 0)

# Q7.III
# pruning
cv_boston <- cv.tree(tree_boston)
cv_boston$dev # 4380.849  4544.815  5601.055  6171.917  6919.608 10419.472 19630.870
cv_boston$size # 7 6 5 4 3 2 1
plot(cv_boston$size, cv_boston$dev, type = "b")

# Q7.IV
prune_boston <- prune.tree(tree_boston, best = 5)
plot(prune_boston)
text(prune_boston, pretty = 0)

# Q7.V
# comparing test errors between whole tree and pruned tree
whole_pred <- predict(tree_boston, newdata = Boston[-train, ])
boston_test <- Boston[-train, "medv"]
mean((whole_pred - boston_test)^2)
# 35.28688
prune_pred <- predict(prune_boston, newdata = Boston[-train, ])
boston_test <- Boston[-train, "medv"]
mean((prune_pred - boston_test)^2)
# 35.90102

############## Problem 8 #############
# regression trees
rm(list=ls())
attach(Carseats)

# Q8.I
tree_carseats <- tree(Sales ~ ., Carseats)
summary(tree_carseats)
par(mfrow = c(1,1))
plot(tree_carseats)
text(tree_carseats, pretty = 0)

# Q8.II
# training data
set.seed(1)
train <- sample(1:nrow(Carseats), nrow(Carseats) / 2)
tree_carseats <- tree(Sales ~ ., Carseats, subset = train)
summary(tree_carseats)
plot(tree_carseats)
text(tree_carseats, pretty = 0)

# Q8.VI
# pruning
cv_carseats <- cv.tree(tree_carseats)
cv_carseats$dev 
# [1]  831.3437  852.3639  868.6815  862.3400  862.3400  893.4641  911.2580  950.2691
# [9]  955.2535 1039.1241 1066.6899 1125.0894 1205.5806 1273.2889 1302.8607 1349.9273
# [17] 1620.4687
cv_carseats$size # 18 17 16 15 14 13 12 11 10  8  7  6  5  4  3  2  1
plot(cv_carseats$size, cv_carseats$dev, type = "b")

prune_carseats <- prune.tree(tree_carseats, best = 5)
plot(prune_carseats)
text(prune_carseats, pretty = 0)
summary(prune_carseats)

# Q8.VII
# comparing test errors between whole tree and pruned tree
carseats_test <- Carseats[-train, "Sales"]
full_pred <- predict(tree_carseats, newdata = Carseats[-train, ])
mean((full_pred - carseats_test)^2)
# 4.922039
prune_pred <- predict(prune_carseats, newdata = Carseats[-train, ])
mean((prune_pred - carseats_test)^2)
# 5.186482

#################### Problem 9 ######################
# various tree methods
rm(list=ls())
library(randomForest)
attach(Boston)

# Q9.I
# a)
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
# bagging means setting the mtry to the number of variables so that all are considered
bag_boston <- randomForest(medv ~ ., data = Boston, subset = train, mtry = 12, importance = TRUE)
boston_test <- Boston[-train, "medv"]
bag_pred <- predict(bag_boston, newdata = Boston[-train, ])
mean((bag_pred - boston_test)^2)
# 23.40359
# b)
bag_boston <- randomForest(medv ~ ., data = Boston, subset = train, mtry = 12, ntree = 25)
bag_pred <- predict(bag_boston, newdata = Boston[-train, ])
mean((bag_pred - boston_test)^2)
# 24.59162
# c)
# this is now RF since we use a smaller number for mtry
rf_boston <- randomForest(medv ~ ., data = Boston, subset = train, mtry = 6, importance = TRUE)
bag_pred <- predict(rf_boston, newdata = Boston[-train, ])
mean((bag_pred - boston_test)^2)
# 20.17751
# d)
importance(bag_boston)
varImpPlot(bag_boston)
importance(rf_boston)
varImpPlot(rf_boston)
# rm and lstat are most important

# Q9.II
# boosting
library(gbm)
set.seed(1)
# a)
boost_boston <- gbm(medv ~ ., data = Boston[train, ],
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4)
summary(boost_boston)
par(mfrow = c(1, 2))
plot(boost_boston, i = "rm")
plot(boost_boston, i = "lstat")
# b)
boost_pred <- predict(boost_boston,
                      newdata = Boston[-train, ], n.trees = 5000)
mean((boost_pred - boston_test)^2)
# 18.39057
# c)
boost_boston <- gbm(medv ~ ., data = Boston[train, ],
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4, shrinkage = 0.2)
summary(boost_boston)
plot(boost_boston, i = "rm")
plot(boost_boston, i = "lstat")
boost_pred <- predict(boost_boston,
                      newdata = Boston[-train, ], n.trees = 5000)
mean((boost_pred - boston_test)^2)
# 16.54778

# Q9.III
# BART
library(BART)

x <- Boston[, 1:12]
y <- Boston[, "medv"]
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[-train, ]
ytest <- y[-train]
set.seed(1)
bartfit <- gbart(xtrain, ytrain, x.test = xtest)
bart_pred <- bartfit$yhat.test.mean
mean((ytest - bart_pred)^2)
# 15.94718

#################### Problem 10 ##################
rm(list=ls())
attach(Carseats)

set.seed(1)
train <- sample(1:nrow(Carseats), nrow(Carseats) / 2)
test <- Carseats[-train, "Sales"]

bag_carseats <- randomForest(Sales ~ ., data = Carseats, subset = train, mtry = 10, importance = TRUE)
bag_pred <- predict(bag_carseats, newdata = Carseats[-train, ])
mean((bag_pred - test)^2)
# 2.623527
# b)
bag_carseats <- randomForest(Sales ~ ., data = Carseats, subset = train, mtry = 10, ntree = 25)
bag_pred <- predict(bag_carseats, newdata = Carseats[-train, ])
mean((bag_pred - test)^2)
# 2.87522
# c)
rf_carseats <- randomForest(Sales ~ ., data = Carseats, subset = train, mtry = 4, importance = TRUE)
bag_pred <- predict(rf_carseats, newdata = Carseats[-train, ])
mean((bag_pred - test)^2)
# 2.819087
# d)
importance(bag_carseats)
varImpPlot(bag_carseats)
importance(rf_carseats)
varImpPlot(rf_carseats)

# Q10.II
# boosting
# a)
boost_carseats <- gbm(Sales ~ ., data = Carseats[train, ],
                    distribution = "gaussian", n.trees = 1000,
                    interaction.depth = 4)
summary(boost_carseats)
plot(boost_carseats, i = "Price")
# b)
boost_pred <- predict(boost_carseats,
                      newdata = Carseats[-train, ], n.trees = 1000)
mean((boost_pred - test)^2)
# 1.899762
# c)
boost_carseats <- gbm(Sales ~ ., data = Carseats[train, ],
                    distribution = "gaussian", n.trees = 1000,
                    interaction.depth = 4, shrinkage = 0.1)
summary(boost_carseats)
plot(boost_carseats, i = "Price")
boost_pred <- predict(boost_carseats,
                      newdata = Carseats[-train, ], n.trees = 1000)
mean((boost_pred - test)^2)
# 2.065894

# Q10.III
# BART
library(BART)

x <- Carseats[, 1:10]
y <- Carseats[, "Sales"]
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[-train, ]
ytest <- y[-train]
set.seed(1)
bartfit <- gbart(xtrain, ytrain, x.test = xtest)
bart_pred <- bartfit$yhat.test.mean
mean((ytest - bart_pred)^2)
# 0.184202

