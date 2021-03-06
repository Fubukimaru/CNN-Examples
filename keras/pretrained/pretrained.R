#' ---
#' jupyter:
#'   jupytext:
#'     formats: ipynb,Rmd:rmarkdown,R:spin
#'     text_representation:
#'       extension: .R
#'       format_name: spin
#'       format_version: '1.0'
#'       jupytext_version: 0.8.6
#'   kernelspec:
#'     display_name: R
#'     language: R
#'     name: ir
#' ---

#' # Installation details

#' Mind that if you want to use your GPU (Nvida CUDA compatible is needed), you
#' need to install the GPU version of the package along with CUDA, Nvidia's package
#' for interfacing between the user and the GPU, and cuDNN, a Deep Neural Network
#' base package.
#'
#' - CUDA: http://www.nvidia.es/object/cuda-parallel-computing-es.html
#' - cuDNN: https://developer.nvidia.com/cudnn

#' Notice that training NN with CPU is slower than using GPU, even though
#' Keras/Tensorflow does some kind of paralelization of the operations using BLAS 
#' libraries (Basic Linear Algebra Subprograms). Therefore, GPU usage is advised 
#' for this session. But don't worry, the code for CPU and GPU is exactly the 
#' same except for one line where we define the execution.

#' # Loading the model

library(keras)
library(imager)

#' Loading Batch-Normalized Inception network
#'
#' Reference:   
#'  - Batch normalization: Accelerating deep network training by reducing internal covariate shift(Loffe et al., 2015).
#'  
#' Based on Inceptionv3 network: rethinking the inception architecture for computer vision (Szegedy et al., 2015) https://arxiv.org/abs/1512.00567

model <- application_inception_v3(include_top = TRUE, weights = "imagenet",
  input_tensor = NULL, input_shape = c(299,299,3), pooling = NULL,
  classes = 1000)

#' Now we check the model:

model

#' # Load model's classes

#' The predictions of the network can be decoded with *imagenet_decode_predictions*, however we will use the original text file so we know how it is done.

synsets <- readLines("synset.txt")
length(synsets)
head(synsets)

#' # Image preprocess functions

# Image preprocess
preproc.image <- function(im, crop = TRUE, dims=3) {
  if (crop) {
    ## Crop the image so it gets same height and width
    shape <- dim(im)
    short.edge <- min(shape[1:2]) # Get the shorter edge from the picture
    # Calculate how much we should crop for each axis
    xx <- floor((shape[1] - short.edge) / 2) 
    yy <- floor((shape[2] - short.edge) / 2)
    im <- crop.borders(im, xx, yy) # Cropped image
  } 
  # Resize to 299 x 299, needed by input of the model.
  resized <- resize(im, 299, 299)
  # Convert to array (x, y, channel)
  arr <- (as.array(resized) - 0.5)*2 # Pixels between -1 and 1 (This is the normalization used in training)
  dim(arr) <- c(299, 299, 3)

  # Reshape to format needed by the network (num, width, height, channel)
  dim(arr) <- c(1,299, 299, 3)
  return(arr)
}

# Result printing
printClassRank <- function(prob, labels, nRes = 10) {
  nRes <- min(nRes, length(labels))
  o <- order(prob, decreasing=TRUE)
  res <- data.frame(class=synsets[o], probability=prob[o])
  head(res, n = nRes)
}

#' # Image classification

# Modifying plot area so that we can see the original photo and the preprocessed
#  photo
oldpar <- par() # We can save the old configuration using empty par()
par(mfrow=c(1,2))

#' ## Starters: Take an image from imageR package - Give me macaws!

im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)

dim(im)

#' 768x512 resolution with 3 channels (RGB), but this network was trained with images of size 299x299, we need to resize them.

preproc <- preproc.image(im)
plot(as.cimg(preproc[,,,]))
dim(preproc)

#' We can observe that the image has been reduced, now has height = width and the image is cropped. Values are now between -1 and 1.

str(preproc)

#' Predict the parrots! In other words, get the class probabilities

prob <- predict(model, preproc)
dim(prob)

#' We get 1000 probabilities,  one for each class. Now we check which are the most representative.

printClassRank(prob, synsets)

#' It's actually not a parrot but a Macaw and it gets correctly predicted!

#' ## Now with something completely different: A laptop with Alpha Channel

im2 <- load.image("images/laptop.png")
plot(im2)
dim(im2)

#'  572x430 with 4 channels... something is wrong.
#'  PNG format allows the image to have transparency using the well-known alpha
#'   channel. This channel marks how much transparency we need to use for each 
#'   pixel. We can remove it using rm.alpha function.

im2 <- rm.alpha(im2)
plot(im2)
dim(im2)


#' Now we have 3 channels, we can proceed as before.

preproc <- preproc.image(im2)
plot(as.cimg(preproc[,,,]))

prob <- predict(model, preproc)
printClassRank(prob, synsets)

#' Monitor, screen, desktop computer, laptop. Not bad.

#' ## Picture proportion is important: Is the network aware of dinosaurs?

im3 <- load.image("images/dinosaur.jpg")
plot(im3)

preproc <- preproc.image(im3)
plot(as.cimg(preproc[,,,]))
prob <- predict(model, preproc)

#' The head gets cut, does it matter?

printClassRank(prob, synsets)

#' African Chamaleon. Almost!

#'  What happens if we resize without cropping the image so that the head is 
#'   preserved?

preproc <- preproc.image(im3, crop=FALSE)
plot(as.cimg(preproc[,,,]))
prob <- predict(model, preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# Frilled lizard. The prediction changed, so the proportions of the image are important.

#' What if we center it manually?

im3.2 <- load.image("images/dinosaurSquare.jpg")
plot(im3.2)

preproc <- preproc.image(im3.2)
plot(as.cimg(preproc[1,,,]))
prob <- predict(model, preproc)

printClassRank(prob, synsets)

#' African Chamaleon still is the most representative 

#' ## Can the network identify Agbar Tower?

im4 <- load.image("images/agbar.jpg")
plot(im4)

preproc <- preproc.image(im4)
plot(as.cimg(preproc[1,,,]))
prob <- predict(model, preproc)

printClassRank(prob, synsets)

#' Spindle, sock... Almost. The network is not quite sure about what Agbar tower is.
#' We should ignore the classification, as the higher probability is very small. Maybe the colors are fooling the network, let's try without color.

preproc <- preproc.image(im4)
preproc <- preproc[1,,,c(2,2,2), drop=F] # Grayscale (setting all the channels to the same value)
plot(as.cimg(preproc[1,,,]))
prob <- predict(model, preproc[1,,,c(1,1,1), drop=F])
printClassRank(prob, synsets)

#' It's not the color, it gets classified as a saxophone.

#' ## Agbar 2: Let's try with another photo

im5 <- load.image("images/Agbar2.png")
plot(im5)

preproc <- preproc.image(im5)
plot(as.cimg(preproc[1,,,]))
prob <- predict(model, preproc)

printClassRank(prob, synsets)

#' The network is still not clear about what is that photo. Probably the bigest lampshade on planet Earth.

#' ## Identifying by aesthetics: You may know what is this one

im6 <- load.image("images/NES.png")
dim(im6)
# Notice that this image has 4 channels!
plot(im6)

# Remove Alpha Channel
im6 <- rm.alpha(im6)

preproc <- preproc.image(im6)
plot(as.cimg(preproc[,,,]))
prob <- predict(model, preproc)

printClassRank(prob, synsets)

#'  Modem, cassette player, CRT screen... An hypothesis on this is that it is recognizing the 
#'   aesthetics of that time this console was built.
