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



