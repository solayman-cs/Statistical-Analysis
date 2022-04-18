

## This example is taken from https://www.machinelearningplus.com/julia/logistic-regression-in-julia-practical-guide-with-examples/

using CSV, Plots,GLM, DataFrames
@rlibrary rpart
#@rlibrary partykit
@rlibrary ggplot2
@rlibrary rattle

# For dataset: https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv

df=DataFrame(CSV.File("Churn_Modelling.csv"))
describe(df)

names(df)
first(df,3) 
# unbalanced : need to fix this for prediction later
sum(df.Exited), length(df.Exited)-sum(df.Exited)

# model
select!(df, Not([:RowNumber, :CustomerId,:Surname]))
first(df,5) |> pretty

# train and test dataset

n, p=size(df)

using Random, Statistics
Random.seed!(23123)

ind=randperm(n)
traindf= df[1:6000,:]
testdf=df[6001:10000,:]

# Model fitting on Train dataset
fm = @formula(Exited ~ CreditScore + Age + Tenure + Balance + NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary + Geography)

logit = GLM.glm(fm, traindf, Binomial(), LogitLink())

treemodel=rpart(fm,traindf)
fancyRpartPlot(treemodel)

#
plotcp(treemodel)
printcp(treemodel)


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



