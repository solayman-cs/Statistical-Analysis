using RCall, RDatasets, DataFrames, CSV

@rlibrary rpart
@rlibrary ggplot2


spdata = CSV.read("spam7.csv", DataFrame)
describe(spdata)

ggplot(spdata,aes(x=:make,y=:dollar)) + geom_line()

using GLM
m1=@formula(yesno~dollar+bang+money+n000+make)
tree= rpart(m1, data=spdata)
printcp(tree)
plotcp(tree)

#@rimport rpart.plot as rplot
@rimport base as rbase
@rlibrary rattle


fancyRpartPlot(tree)
