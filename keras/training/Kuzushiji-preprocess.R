# -*- coding: utf-8 -*-
#' ---
#' jupyter:
#'   jupytext:
#'     formats: ipynb,Rmd:rmarkdown, R
#'     text_representation:
#'       extension: .R
#'       format_name: spin
#'       format_version: '1.0'
#'       jupytext_version: 0.8.5
#'   kernelspec:
#'     display_name: R
#'     language: R
#'     name: ir
#'   language_info:
#'     codemirror_mode: r
#'     file_extension: .r
#'     mimetype: text/x-r-source
#'     name: R
#'     pygments_lexer: r
#'     version: 3.5.1
#' ---

library(R.utils)
library(keras)  # FashionMNIST dataset
library(nnet)  # Neural networks

#' # Dataset

#' We're going to use the Kuzushiji dataset which is a dataset that 
#' contains 60000 training images and 10000 testing images in grayscale (one 
#' channel) and of size 28x28. Kuzushiji comes in MNIST original format (packed byte-encoded images), so we need a special function to read it.

gunzip('kuzushiji/train-images-idx3-ubyte.gz')
gunzip('kuzushiji/t10k-images-idx3-ubyte.gz')
gunzip('kuzushiji/train-labels-idx1-ubyte.gz')
gunzip('kuzushiji/t10k-labels-idx1-ubyte.gz')

#' Now we define some auxiliary functions for loading this dataset.
#' The dataset comes in a binary form. Details about the format can be found on
#' Yann LeCun's website http://yann.lecun.com/exdb/mnist/

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

#' Loading train and test images

train <- load_image_file('kuzushiji/train-images-idx3-ubyte')
test <- load_image_file('kuzushiji/t10k-images-idx3-ubyte')

#' Loading labels

train$y <- load_label_file('kuzushiji/train-labels-idx1-ubyte')
test$y <- load_label_file('kuzushiji/t10k-labels-idx1-ubyte') 

#' We get the following structure:
#'
#' - train: Training dataset
#'     + x: the predictors, 28x28 pixels image in grayscale.
#'     + y: the response
#' - test: Testing datset (with x and y)
#'
#' We can see the images with the following function:

str(train)

#' For CNNs we need to have them in 28x28 format instead of an array of 784 pixels

dim(train$x) <- c(dim(train$x)[1], 28,28)

dim(test$x) <- c(dim(test$x)[1], 28,28)

#' ## Response reencode

#' Notice that in y we have an integer from 0 to 9 (10 classes). They are in fact the following:
#' - 0: お - o
#' - 1: き - ki 
#' - 2: す - su 
#' - 3: つ - tsu
#' - 4: な - na
#' - 5: は - ha
#' - 6: ま - ma
#' - 7: や - ya
#' - 8: れ - re 
#' - 9: を - wo
#'
#' We recode the response variable to factor.

classmap <- read.csv("kuzushiji/kmnist_classmap.csv")

classmap$romaji <- c("o","ki","su","tsu","na","ha","ma","ya","re","wo")
classmap

#' We use the romaji codification just to prevent problems with UTF8 characters

classString <- classmap$romaji
# y+1 because 0 is the first class and in R we start indexing at 1!
train$yFactor <- as.factor(classString[train$y+1]) 
test$yFactor <- as.factor(classString[test$y+1])

#' For the CNN we use one hot encoding to produce a vector of 10 values per sample, with a one on the class (probablity of belonging to a given class).

train$yOneHot <- class.ind(train$yFactor)
test$yOneHot <- class.ind(test$yFactor)

#' *class.ind* reorders the classes alfabetically, therefore we need to revert this order to the original provided. We use *match* over the column names to get a vector of the reorder to match the column names to **classString**.

colnames(train$yOneHot)
classString
(m <- match(classString, colnames(test$yOneHot))) 

train$yOneHot <- train$yOneHot[,m]
test$yOneHot <- test$yOneHot[,m]

#' Now the order is correct

colnames(train$yOneHot)
classString

str(train$yOneHot)
train$yFactor[1:10]
train$yOneHot[1:10,]

#' ## Add missing dimension

#' Convolutional layers will expect the input to have 4 dimensions:
#' - Sample dimension
#' - Height dimension
#' - Width dimension
#' - Channel dimension
#'
#' In our case we have only one channel as the image is grayscale. If it's a color image we would have 3 or 4 channels (Red, Green, Blue and Alpha (transparency)). We need to add the missing dimension, however this will not modify the data. 

dim(train$x) <- c(dim(train$x),1)
dim(test$x) <- c(dim(test$x),1)

#' ## dataset visualization

#' By the dataset organization, the elements are inverted. Just for display we flip them using `lim = rev(range(0,1))`

rotate <- function(x) apply(x, 2, rev)
show_image <- function(imgarray, col=gray(12:1/12), ...) {
  image((matrix(imgarray, nrow=28)), col=col, ylim = rev(range(0,1)), ...)
}

show_image(train$x[14,,,])
train$yFactor[14] # This ha is katakana! ハ
show_image(train$x[416,,,])
train$yFactor[416]

#' ## Create a dataset for nnet

#' Now we prepare join the X and the Y in a data.frame.

nnetData <- data.frame(train$x, class=train$yFactor)

nnetDataTest <- data.frame(test$x, class=test$yFactor)

#' # Your code should be here
