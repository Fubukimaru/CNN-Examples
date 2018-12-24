
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
library(caret)
# model.nnet <- nnet(x=train$x, y=class.ind(train$yFactor), softmax=TRUE, size=50, maxit=300, decay=0.5, MaxNWts = 39760)

## specify 10x10 CV
trc <- trainControl (method="repeatedcv", number=2, repeats=1)

(decays <- 10^seq(-3,0,by=0.25))

library(doMC)
# Use all cores except one (recommended if you want to use your computer for something else)

cores <- min(detectCores()-1, ceiling(length(decays)/2))
registerDoMC(cores = cores) 
nnetData <- data.frame(train$x, class=train$yFactor)

print("Executing training")
## WARNING: this takes some time (around 10')
model.2x1CV <- train (class ~ ., data=nnetData, method='nnet', maxit = 300, trace = FALSE,
                      tuneGrid = expand.grid(.size=50,.decay=decays), trControl=trc, MaxNWts=39760)


save(model.2x1CV, file="nnet.mod")

## ------------------------------------------------------------------------
load("nnet.mod")

