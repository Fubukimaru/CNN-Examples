# Load the Fashion MNIST digit recognition dataset into R
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# call:  show_digit(train$x[5,])   to see a digit.
# Slightly modified (mostly name changes) version of:
# brendan o'connor - gist.github.com/39760 - anyall.org
load_fashion_mnist <- function() {
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
  train <<- load_image_file('~/data/fashionMNIST/train-images-idx3-ubyte')
  test <<- load_image_file('~/data/fashionMNIST/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('~/data/fashionMNIST/train-labels-idx1-ubyte')
  test$y <<- load_label_file('~/data/fashionMNIST/t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}