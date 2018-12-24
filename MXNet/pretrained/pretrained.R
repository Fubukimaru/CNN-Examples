################################################################################
#                           INSTALLATION DETAILS                               #
################################################################################

 
# MXNet is a package that is quite new in the R world, therefore is not included
# in CRAN repository. In order to install this package you can follow the instructions
# found in https://mxnet.incubator.apache.org/install/index.html. Windows/MacOS is
# recommended for starting to use this package as installation is very simple. 
# The GPU version is not available out of the box on MacOS and may require to follow
# Linux installation procedure.
# 
# Mind that if you want to use your GPU (Nvida CUDA compatible is needed), you
# need to install the GPU version of the package along with CUDA, Nvidia's package
# for interfacing between the user and the GPU, and cuDNN, a Deep Neural Network
# base package.
# 
# - CUDA: http://www.nvidia.es/object/cuda-parallel-computing-es.html
# - cuDNN: https://developer.nvidia.com/cudnn

# 
# Notice that training NN with CPU is slower than using GPU, even though MXNet does
# some kind of paralelization of the operations using BLAS libraries (Basic Linear
# Algebra Subprograms). Therefore, GPU usage is advised for this session. But don't
# worry, the code for CPU and GPU is exactly the same except for one line where
# we define the execution.



################################################################################
#                               LOADING MODEL                                  #
################################################################################

library(mxnet)
library(imager)

# Loading Batch-Normalized Inception network

# Reference:   
#  Batch normalization: Accelerating deep network training by reducing internal 
#   covariate shift(Loffe et al., 2015).

# Based on:
# Inceptionv3 network: rethinking the inception architecture for computer vision
# (Szegedy et al., 2015) https://arxiv.org/abs/1512.00567

model = mx.model.load("Inception/Inception_BN", iteration=39)

# Load average image (of all the imaged used for training)
mean.img <- as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
plot(as.cimg(mean.img))

# Now we get the class names
synsets <- readLines("Inception/synset.txt")
length(synsets)
head(synsets)



################################################################################
#                       USEFULL AUXILIARY FUNCTIONS                            #
################################################################################

# Image preprocess
preproc.image <- function(im, mean.image, crop = TRUE, dims=3) {
  if (crop) {
    ## Crop the image so it gets same height and width
    shape <- dim(im)
    short.edge <- min(shape[1:2]) # Get the shorter edge from the picture
    # Calculate how much we should crop for each axis
    xx <- floor((shape[1] - short.edge) / 2) 
    yy <- floor((shape[2] - short.edge) / 2)
    im <- crop.borders(im, xx, yy) # Cropped image
  } 
  # Resize to 224 x 224, needed by input of the model.
  resized <- resize(im, 224, 224)
  # Convert to array (x, y, channel)
  arr <- as.array(resized) * 255 # From 0..1 to 0..255 (RGB integer codification)
  dim(arr) <- c(224, 224, 3)
  # Subtract the mean
  preproc <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(preproc) <- c(224, 224, 3, 1)
  return(preproc)
}

# Result printing
printClassRank <- function(prob, labels, nRes = 10) {
  nRes <- min(nRes, length(labels))
  o <- order(prob, decreasing=TRUE)
  res <- data.frame(class=synsets[o], probability=prob[o])
  head(res, n = nRes)
}

################################################################################
#                           IMAGE CLASSIFICATION                               # 
################################################################################

# Modifying plot area so that we can see the original photo and the preprocessed
#  photo
oldpar <- par() # We can save the old configuration using empty par()
par(mfrow=c(1,2))




######## Starters: Take an image from imageR package - Give me macaws! #########
im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)

dim(im)
# 768x512 resolution with 3 channels (RGB), but this network was trained with
#  images of size 224x224, we need to resize them.

# Preprocess the parrots
preproc <- preproc.image(im, mean.img)
plot(as.cimg(preproc[,,,1]))
dim(preproc)
# We can observe that the image has been reduced, now has height = width and
#  the color is a little off because of the subtraction of the mean image.

# Predict the parrots! In other words, get the class probabilities
prob <- predict(model, X=preproc)
dim(prob)

# Which classes ar the most representative
printClassRank(prob, synsets)
# It's a Macaw!




##### Now with something completely different: A laptop with Alpha Channel #####
# Load new image
im2 <- load.image("images/laptop.png")
plot(im2)
dim(im2)

# 572x430 with 4 channels... something is wrong.
# PNG format allows the image to have transparency using the well-known alpha
#  channel. This channel marks how much transparency we need to use for each 
#  pixel. We can remove it using rm.alpha function.

# Remove Alpha Channel
im2 <- rm.alpha(im2)
plot(im2)
dim(im2)
# Now we have 3 channels, we can proceed as before.


# Preprocess the laptop
preproc <- preproc.image(im2, mean.img)
plot(im2); plot(as.cimg(preproc[,,,1]))

# Prediction again...
prob <- predict(model, X=preproc)
printClassRank(prob, synsets)
# Notebook, laptop, monitor. Very good!




###### Picture proportion is important: Is the network aware of dinosaurs?######
# Load new image
im3 <- load.image("images/dinosaur.jpg")
plot(im3)

preproc <- preproc.image(im3, mean.img)
plot(as.cimg(preproc[,,,1]))
# The head gets cut, does it matter?
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# African Chamaleon. Almost!

# What happens if we resize without cropping the image so that the head is 
#  preserved?
preproc <- preproc.image(im3, mean.img, crop=FALSE)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# Ibex. The proportions of the image look like important.

# What if we center it manually?
im3.2 <- load.image("images/dinosaurSquare.jpg")
plot(im3.2)

preproc <- preproc.image(im3.2, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# African Chamaleon still is the most representative 




################# Can the network identify Agbar Tower? ########################
# Load new image
im4 <- load.image("images/agbar.jpg")
plot(im4)

preproc <- preproc.image(im4, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# Mitten... Almost. The network is not quite sure about what Agbar tower is.
# We should ignore classification, as the higher probability is very small.

# Let's try with another photo
im5 <- load.image("images/Agbar2.png")
plot(im5)

# Normalize the laptop
preproc <- preproc.image(im5, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

printClassRank(prob, synsets)
# Waterbottle. Closer.



########## Identifying by aesthetics; You may know what is this one ############
im6 <- load.image("images/NES.png")
dim(im6)
# Notice that this image has 4 channels!
plot(im6)

# Remove Alpha Channel
im6 <- rm.alpha(im6)

preproc <- preproc.image(im6, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

printClassRank(prob, synsets)
# Modem, Tape player... An hypothesis on this is that it  is recognizing the 
#  aesthetics of that time.
