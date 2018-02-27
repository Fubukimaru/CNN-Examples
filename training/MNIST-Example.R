# https://mxnet.incubator.apache.org/tutorials/r/mnistCompetition.html

require(mxnet)


train <- read.csv('~/data/MNIST/train.csv', header=TRUE)
test <- read.csv('~/data/MNIST/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]


train.x <- t(train.x/255)
test <- t(test/255)

table(train.y)


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")


devices <- mx.cpu()

mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# Start training with 1 devices
# [1] Train-accuracy=0.859832935560859
# [2] Train-accuracy=0.958309523809525
# [3] Train-accuracy=0.970809523809526
# [4] Train-accuracy=0.977452380952384
# [5] Train-accuracy=0.983000000000003
# [6] Train-accuracy=0.98595238095238




# Lenet

#input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)


train.array <- array(train.x, c(28, 28, 1, ncol(train.x)))
test.array <- array(test, c(28, 28, 1, ncol(test)))


device.cpu <- mx.cpu()

logger <- mx.metric.logger$new()

mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=3, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(10, logger))

# [1] Train-accuracy=0.553532219570405
# [2] Train-accuracy=0.969285714285718
# [3] Train-accuracy=0.981428571428575
# [4] Train-accuracy=0.988333333333336
# [5] Train-accuracy=0.99161904761905h

# Test prediction
preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)