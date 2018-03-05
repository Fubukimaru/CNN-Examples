library(mxnet)

# Input preprocessing
# ===================
# 
# Before starting we need to process the images, as they are not a standard 
# data.frame. We first are required to build a list of filenames out of the images.
# The files are composed as binaries with a custom format in order to save space.
# 
# First we decompress the files
## ------------------------------------------------------------------------
  
  unzip('fashionMNIST/train-images-idx3-ubyte.zip', exdir = "fashionMNIST")
  unzip('fashionMNIST/t10k-images-idx3-ubyte.zip', exdir = "fashionMNIST")
  unzip('fashionMNIST/train-labels-idx1-ubyte.zip', exdir = "fashionMNIST")
  unzip('fashionMNIST/t10k-labels-idx1-ubyte.zip', exdir = "fashionMNIST")

## ------------------------------------------------------------------------
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }

  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  
  train <- load_image_file('fashionMNIST/train-images-idx3-ubyte')
  test <- load_image_file('fashionMNIST/t10k-images-idx3-ubyte')
  
  train$y <- load_label_file('fashionMNIST/train-labels-idx1-ubyte')
  test$y <- load_label_file('fashionMNIST/t10k-labels-idx1-ubyte') 
  
  
  classString <- c("T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal",
                  "Shirt","Sneaker", "Bag","Ankle boot")
  
  train$yFactor <- as.factor(classString[train$y+1])
  test$yFactor <- as.factor(classString[test$y+1])


# Testing with nnet
# -----------------
library(nnet)

model.nnet <- nnet(x=train$x, y=class.ind(train$yFactor), softmax=TRUE, size=50, maxit=300, decay=0.5, MaxNWts = 39760)

save(model.nnet, file="nnet.mod")

## ------------------------------------------------------------------------
load("nnet.mod")
p1 <- as.factor(predict (model.nnet, type="class"))

# +1 because indexing in R starts at 1, not at 0!
(t1 <- table(Truth=train$yFactor, Pred=p1))
(1-sum(diag(t1))/sum(t1))*100



# Reshape data for CNN input
# --------------------------
# 
# As input data we have now a matrix of N rows times M columns. The rows represent
# each image and the columns each pixel of the image. We will reshape it to take
# the shape of a 4 dimensional array. First identifier will be the image, second 
# the channel and third and fourth will be the coordinates of a given pixel of the
# image, i.e. Width x Height x Channels x Images.
train$x <- array(train$x, c(train$n,1,28,28)) # Transform into Nx1x28x28 array
train$x <- aperm(train$x, c(3,4,2,1))         # Permutate columns into 28x28x1xN array
test$x <- array(test$x, c(test$n,1,28,28))
test$x <- aperm(test$x, c(3,4,2,1))

# 
# Now we divide between training set and validation set. Mind that as we are working 
# with a grayscale image we only have one channel. This is dangerous in R as if we
# subset the dataset using [] operator, R will remove the dimension that only has one identifier (channels). To prevent this we have to do as follows: x[i, j, drop = FALSE] instead of x[i,j].
# 
valSamp <- sample(length(train$y)/3)

validation <- list(x = train$x[,,,valSamp, drop=FALSE], y= train$y[valSamp]) # drop=FALSE forbids R to drop dimensions with length 1
train$x <- train$x[,,,-valSamp, drop=FALSE]
train$y <- train$y[-valSamp]



# Check images after preprocess
# -----------------------------
show_image <- function(imgarray, col=gray(12:1/12), ...) {
  image(matrix(imgarray, nrow=28)[,28:1], col=col, ...)
}

show_image(train$x[,,,13])
show_image(train$x[,,,54])


# Model architecture definition
# -----------------------------
# 
# Now we have to define the CNN arcir <- rbind(iris3[,,1],iris3[,,2],iris3[,,3])
# targets <- class.ind( c(rep("s", 50), rep("c", 50), rep("v", 50)) )
# samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))
# ir1 <- nnet(ir[samp,], targets[samp,], size = 2, rang = 0.1,
# decay = 5e-4, maxit = 200)hitecture. In this case we use LeNet, proposed
# by LeCun et al. (Gradient-based learning applied to document recognition. 
# Proceedings of the IEEE, november 1998). 
# 
# It is composed by two packs of convolutional-activation(tanh)-pooling layers and
# two fully connected layers with a softmax at the end.
# 
# In MXNet, as in most of the packages, we define layers as symbols and the 
# connections between those symbols using the data parameter.


# LeNet
# -----

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

# And now we an visualize the model representation with the following commands:
graph.viz(lenet)



# Model training
# --------------
# 
# Now we're going to train the network using CPU. Mind that if you want to use
# GPUs you need to have the GPU version of the package and the required Nvidia
# packages.
# 
## ------------------------------------------------------------------------
devices <- mx.cpu()
# devices <- mx.gpu() # Use GPU

mx.set.seed(123)
model_mxnet <- mx.model.FeedForward.create(lenet,
                X=train$x, 
                y=train$y,
                eval.data = list(data=validation$x, label=validation$y), # Data for epoch evaluation
                array.batch.size = 100, # Size of each training batch 
                ctx=devices, # Which device to use for training 
                num.round=5,
                learning.rate=0.05, # Stochastic Gradient Descent learning rate
                wd=0.001, # Weight decay
                momentum=0.8,
                clip_gradient=1, # Gradient clip. Keep gradient between -1 and 1
                eval.metric=mx.metric.accuracy, 
                epoch.end.callback = mx.callback.log.train.metric(1))


save(model_mxnet, file="mxnet.mod")
load("mxnet.mod")
# Start training with 1 devices
# [1] Train-accuracy=0.679799498746868
# [1] Validation-accuracy=0.7516
# [2] Train-accuracy=0.770025
# [2] Validation-accuracy=0.7774
# [3] Train-accuracy=0.801225
# [3] Validation-accuracy=0.79505
# [4] Train-accuracy=0.816700000000001
# [4] Validation-accuracy=0.80725
# [5] Train-accuracy=0.819
# [5] Validation-accuracy=0.78865 



# Generating predictions
# ----------------------

# Predicting the label for the test set
pred_prob<- t(predict(model_mxnet, test$x))
head(pred_prob)
# For each element we get the probability of that element to be of each class,
#  therefore we search for the value that is maximum in each row as follows:
predClass <- factor(apply(pred_prob,1,which.max))
levels(predClass) <- classString # Change integers to string representation

trueClass <- factor(test$y)
levels(trueClass) <- classString

# Now we do a confusion matrix and analyze it
(cMatrix <- table(trueClass,predClass))

# You should get something like this:
#             predClass
#trueClass     T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag  Ankle boot
#  T-shirt/top         831       5        1     7    4      1   139       0  12           0
#  Trouser              14     949        2    18    8      0     7       0   2           0
#  Pullover             12       0      169     4  215      0   593       0   7           0
#  Dress                44      27        0   789   62      0    78       0   0           0
#  Coat                  1       4       14    28  628      0   324       0   1           0
#  Sandal                0       0        0     1    0    954     0      41   1           3
#  Shirt               179       4       16    21   58      0   705       0  17           0
#  Sneaker               0       0        0     0    0     16     0     975   0           9
#  Bag                   2       1        1     5    4      2    35       4 946           0
#  Ankle boot            0       0        0     0    0     30     0      99   1         870

correctClass <- sum(diag(cMatrix))
total <- sum(cMatrix)
(accuracy <- correctClass/total)

