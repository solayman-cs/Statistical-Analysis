

## This example is taken from https://www.machinelearningplus.com/julia/logistic-regression-in-julia-practical-guide-with-examples/

using CSV, Plots,GLM, DataFrames

using RCall
@rlibrary rpart
#@rlibrary partykit
@rlibrary ggplot2
@rlibrary rattle

# For dataset: https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv

df=DataFrame(CSV.File("Churn_Modelling.csv"))
describe(df)


# model
select!(df, Not([:RowNumber, :CustomerId,:Surname]))
first(df,5)

# train and test dataset

n, p=size(df)

using Random, Statistics, StatsBase
Random.seed!(23123)

#ind=randperm(n)
df1= df[sample(1:10000,6000),:]

#ind1=randperm(n)
df2=df[sample(1:10000,6000),:]
# Model fitting on Train dataset
fm = @formula(Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary + Geography+Gender)

logit1 = GLM.glm(fm, df1, Binomial(), LogitLink())
logit2 = GLM.glm(fm, df2, Binomial(), LogitLink())


treemodel=rpart(fm,df1)
fancyRpartPlot(treemodel)

treemodel2=rpart(fm,df2)
fancyRpartPlot(treemodel2)


#
plotcp(treemodel)
plotcp(treemodel2)



#
printcp(treemodel)
printcp(treemodel2)


#######################################
###### Random forest & Bagging
#######################################
@rlibrary(randomForest)

rfmodel=randomForest(fm , data = df1)
rfmodel


R"plot($rfmodel)"

varImpPlot(rfmodel)

# Bagging
bagmodel=randomForest(fm , data = df1,mtry=10)
bagmodel

Xdf=df1[:,1:10]
Ydf=df1[:,11]

trf=tuneRF(Xdf,Ydf)

prf= R"predict($rfmodel)"

rfcv(Xdf,Ydf)

################################################
### Boosting ############################
##########################################
@rlibrary gbm
using CategoricalArrays

df.Geography=CategoricalArray(df.Geography)
df.Gender=CategoricalArray(df.Gender)


gbmfit=gbm(fm,distribution="bernoulli",data=df,var"n.trees"=1000,var"cv.folds"=5)

names(gbmfit)

sqrt(minimum(gbmfit[Symbol("cv.error")]))

gbm_perf(gbmfit,method="cv")

# tuning more parameters
# creating hyperparameter grid

@rimport base as rbase

hyper_grid = rbase.expand_grid(shrinkage = [.01, .1, .3],  interaction_depth = [1, 3, 5],
  minobsinnode = [5, 10, 15],
  bag_fraction = [.65, .8, 1], 
  optimal_trees = 0.0,               # a place to dump results
  min_RMSE = 0.0                     # a place to dump results
)

rbase.dim(hyper_grid)

for i in 1:rbase.dim(hyper_grid)[1]
    gbm_tune=gbm(fm,distribution="bernoulli",data=df,var"n.trees"=1000,
                    var"interaction.depth"=hyper_grid[:interaction_depth][i],
                    shrinkage=hyper_grid[:shrinkage][i],
                    var"n.minobsinnode"=hyper_grid[:minobsinnode][i],
                    var"bag.fraction"=hyper_grid[:bag_fraction][i],
                    var"train.fraction"=0.6)

        hyper_grid[:optimal_trees][i]= rbase.which_min(gbm_tune[Symbol("valid.error")])

        hyper_grid[:min_RMSE][i]= sqrt(minimum(gbm_tune[Symbol("valid.error")]))
        println(i)
end

## 
minindex=rbase.which_min(hyper_grid[:min_RMSE])

i=convert(Int,minindex)

gbmfitfinal= gbm(fm,distribution="bernoulli",data=df,var"n.trees"=1000,
var"interaction.depth"=hyper_grid[:interaction_depth][i],
shrinkage=hyper_grid[:shrinkage][i],
var"n.minobsinnode"=hyper_grid[:minobsinnode][i],
var"bag.fraction"=hyper_grid[:bag_fraction][i],
var"train.fraction"=0.6)


## variable importance
summary_gbm(gbmfitfinal,cBars=10,method=relative_influence,las=2)


Base.names(gbmfitfinal)

pretty_gbm_tree(gbmfitfinal,var"i.tree"=4)

plot_gbm(gbmfitfinal)

pred=zeros(rbase.dim(df)[1])
@.  pred=Base.exp(gbmfitfinal[:fit]) / (1 + Base.exp(gbmfitfinal[:fit]))

calibrate_plot(df[!,:Exited], pred)

































#########################
################################
ptreemodel=prune(treemodel,cp=0.013)
fancyRpartPlot(ptreemodel)

@rput ptreemodel testdf

R"treetestpred=predict(ptreemodel,testdf)"
#@rget treetestpred
# 


#deviance(logit)
#
tpred=GLM.predict(logit, testdf)

# confusion matrix
tpredbin=ifelse.(tpred .>= 0.5 , 1 , 0)

#using RCall

R"library(ROCR)"

target=testdf.Exited
@rput tpred
@rput target

### use tpred for logistic
### use treetestpred for tree below

R"pred=prediction(treetestpred,target)"
R"pred2=prediction(tpred,target)"

R"""perf=performance(pred,"tpr","fpr")"""
R"""perf2=performance(pred2,"tpr","fpr")"""


## Tree model ROC versus Logistic ROC

R"plot(perf,colorize=TRUE)"
R"plot(perf2, add=TRUE, colorize=FALSE)"



### AUC values  Tree model is better
R""" aucout=performance(pred, measure = "auc")"""
R"auc = aucout@y.values[[1]]"

R""" aucout2=performance(pred2, measure = "auc")"""
R"auc2 = aucout2@y.values[[1]]"



@rget auc


###################################################
######## Instability of Trees ####################





