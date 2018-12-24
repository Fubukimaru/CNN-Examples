library(mxnet)

################################################################################
#                                 PREPROCESS                                   #
################################################################################

# We're going to use the FashionMNIST from Zalando which is a dataset that 
# contains 60000 training images and 10000 testing images in grayscale (one 
# channel) and of size 28x28.

# First we decompress the files
  
unzip('fashionMNIST/train-images-idx3-ubyte.zip', exdir = "fashionMNIST")
unzip('fashionMNIST/t10k-images-idx3-ubyte.zip', exdir = "fashionMNIST")
unzip('fashionMNIST/train-labels-idx1-ubyte.zip', exdir = "fashionMNIST")
unzip('fashionMNIST/t10k-labels-idx1-ubyte.zip', exdir = "fashionMNIST")


# Now we define some auxiliary functions for loading this dataset.
# The dataset comes in a binary form. Details about the format can be found on
# Yann LeCun's website http://yann.lecun.com/exdb/mnist/

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

# Loading train and test images
train <- load_image_file('fashionMNIST/train-images-idx3-ubyte')
test <- load_image_file('fashionMNIST/t10k-images-idx3-ubyte')
  
# Loading labels
train$y <- load_label_file('fashionMNIST/train-labels-idx1-ubyte')
test$y <- load_label_file('fashionMNIST/t10k-labels-idx1-ubyte') 
  
# Create a factor label array
classString <- c("T-shirt/top","Trouser", "Pullover", "Dress", "Coat", "Sandal",
                  "Shirt","Sneaker", "Bag","Ankle boot")
  
train$yFactor <- as.factor(classString[train$y+1])
test$yFactor <- as.factor(classString[test$y+1])


# Now we can check the images check images 
show_image <- function(imgarray, col=gray(12:1/12), ...) {
  image(matrix(imgarray, nrow=28)[,28:1], col=col, ...)
}

oldpar <- par()
par(mfrow=c(3,3))
show_image(train$x[1,])
show_image(train$x[13,])
show_image(train$x[54,])
show_image(train$x[100,])
show_image(train$x[130,])
show_image(train$x[230,])
show_image(train$x[412,])
show_image(train$x[454,])
show_image(train$x[540,])
par(oldpar)




################################################################################
#                           TESTING WITH NNET                                  #
################################################################################

library(nnet)
library(caret)

##################### Training a single simple model ###########################
model.nnet.simple <- nnet(x=train$x, y=class.ind(train$yFactor), softmax=TRUE, 
                          size=10, maxit=100, decay=0.5, MaxNWts = 39760)

save(model.nnet.simple, file="nnet-simple.mod")
load("nnet-simple.mod")

predT <- predict(model.nnet.simple, newdata=test$x, type="class")

(tab <- table(Truth=test$yFactor, Pred=predT))
(sum(diag(tab))/sum(tab))*100

# 74.33% accuracy. Fair enough.


########################### Specify 10x10 CV ###################################
trc <- trainControl (method="repeatedcv", number=2, repeats=1)

(decays <- 10^seq(-3,0,by=0.25))

library(doMC)
# Use all cores except one (recommended if you want to use your computer for something else)

cores <- min(detectCores()-1, ceiling(length(decays)/2))
registerDoMC(cores = cores) 

# Build a data.frame for the process (training will fail if we use it as in the previous nnet model)
nnetData <- data.frame(train$x, class=train$yFactor)

print("Executing training")
## Warning: This takes a while... order of hours!
model.2x1CV <- train (class ~ ., data=nnetData, method='nnet', maxit = 300, trace = FALSE,
                      tuneGrid = expand.grid(.size=50,.decay=decays), trControl=trc, MaxNWts=39760)


save(model.2x1CV, file="nnet-cv.mod")
rm(nnetData) # Not needed anymore...

load("nnet-cv.mod") # This structure is 500MB...!
(model.2x1CV)
#  decay        Accuracy   Kappa      Accuracy SD   Kappa SD
#  0.001000000  0.8259000  0.8065556  0.0043840620  0.0048711800
#  0.001778279  0.8295500  0.8106111  0.0002592725  0.0002880805
#  0.003162278  0.8235167  0.8039074  0.0081788684  0.0090876316
#  0.005623413  0.8219833  0.8022037  0.0011078006  0.0012308896
#  0.010000000  0.8287500  0.8097222  0.0043133514  0.0047926126
#  0.017782794  0.8352167  0.8169074  0.0025220142  0.0028022380
#  0.031622777  0.8358833  0.8176481  0.0006363961  0.0007071068
#  0.056234133  0.8329833  0.8144259  0.0005421152  0.0006023502
#  0.100000000  0.8377833  0.8197593  0.0017677670  0.0019641855
#  0.177827941  0.8252000  0.8057778  0.0063168206  0.0070186895
#  0.316227766  0.8333500  0.8148333  0.0031819805  0.0035355339
#  0.562341325  0.8377333  0.8197037  0.0032526912  0.0036141013
#  1.000000000  0.8394167  0.8215741  0.0064818122  0.0072020135

# So, with 50 hidden units we get around 83% TRAINING accuracy.
# And it takes veeeeery long to compute.

model.nnet <- model.2x1CV$finalModel

# As we trained using a dataframe with names, this step is required.
tmp <- data.frame(test$x)
names(tmp) <- model.nnet$xNames
head(tmp)

predT2 <- as.factor(predict (model.nnet, newdata=tmp, type="class"))

# +1 because indexing in R starts at 1, not at 0!
(t1 <- table(Truth=test$yFactor, Pred=predT2))
#             Pred
# Truth         Ankle boot Bag Coat Dress Pullover Sandal Shirt Sneaker Trouser T-shirt/top
#  Ankle boot         934   1    0     0        0     11     0      54       0            0
#  Bag                  0 947    3     4        6      5    21       6       1            7
#  Coat                 0   6  769    48      102      1    72       0       1            1 
#  Dress                0   6   45   858        4      0    29       0      17           41
#  Pullover             0   4  168    17      708      0    68       0       4           31
#  Sandal              32  10    0     1        0    918     0      38       0            1
#  Shirt                2  27  126    33       91      2   558       0       3          158
#  Sneaker             32   1    0     0        0     34     0     933       0            0
#  Trouser              0   1    7    26        3      0     3       0     955            5
#  T-shirt/top          1  15    4    45       14      1   111       1       4          804

(sum(diag(t1))/sum(t1))*100
# 83.84% of training error. Better than our first attempt!


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




# Model architecture definition
# -----------------------------
# 
# Now we have to define the CNN architecture. In this case we use LeNet, proposed
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
model.mxnet <- mx.model.FeedForward.create(lenet,
                X=train$x, 
                y=train$y,
                eval.data = list(data=validation$x, label=validation$y), # Data for epoch evaluation
                array.batch.size = 100, # Size of each training batch 
                ctx=devices, # Which device to use for training 
                num.round=5,
                learning.rate=0.05, # Stochastic Gradient Descent learning rate
                wd=0.001, # Weight decay
                momentum=0.8, # Accelerator
                clip_gradient=1, # Gradient clip. Keep gradient between -1 and 1
                eval.metric=mx.metric.accuracy, 
                epoch.end.callback = mx.callback.log.train.metric(1))

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
# ...


# MXNet has its own model saving format and functions (same as we used in Inception)
mx.model.save(model.mxnet, "mxnet.mod", 50)

model.mxnet <- mx.model.load("mxnet.mod", iteration=50)

################################################################################
#                        Generating predictions                                #
################################################################################

# Predicting the label for the test set
pred_prob<- t(predict(model.mxnet, test$x))
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
#trueClass     T-shirt/top Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle boot
#  T-shirt/top         879       2       28    16    6      1    59       0   9          0
#  Trouser               4     971        2    13    8      0     1       0   1          0
#  Pullover             13       1      761     9  165      0    48       0   3          0
#  Dress                21       9       12   914   33      0     6       0   5          0
#  Coat                  0       0       54    44  850      0    52       0   0          0
#  Sandal                0       0        0     1    0    970     0      23   1          5
#  Shirt               175       0      113    29  113      0   554       0  16          0
#  Sneaker               0       0        0     0    0     11     0     979   0         10
#  Bag                   4       2       10     5    7      1     2       3 966          0
#  Ankle boot            0       0        1     0    0      6     0      61   1        931

correctClass <- sum(diag(cMatrix))
total <- sum(cMatrix)
(accuracy <- correctClass/total)
# 87.75% accuracy. Better and faster!
