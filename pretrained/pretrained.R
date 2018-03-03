# From this tutorial: https://mxnet.incubator.apache.org/tutorials/r/classifyRealImageWithPretrainedModel.html
# Batch Normalization: https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c

require(mxnet)
require(imager)

# Loading Batch-Normalized Inception network
# Reference: Ioffe, Sergey, and Christian Szegedy. 
#  “Batch normalization: Accelerating deep network training by reducing internal 
#   covariate shift.” arXiv preprint arXiv:1502.03167 (2015).
model = mx.model.load("Inception/Inception_BN", iteration=39)

# Load average image (of all the imaged used for training)
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])
plot(as.cimg(mean.img))


# Image preprocess
preproc.image <- function(im, mean.image, dims=3) {
  ## Crop the image so it gets same height and width
  shape <- dim(im)
  short.edge <- min(shape[1:2]) # Get the shorter edge from the picture
  # Calculate how much we should crop for each axis
  xx <- floor((shape[1] - short.edge) / 2) 
  yy <- floor((shape[2] - short.edge) / 2)
  cropped <- crop.borders(im, xx, yy)
  
  # Resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
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

# Now we get the class names
synsets <- readLines("Inception/synset.txt")

# Modifying plot area so that we can see the original photo and the preprocessed
#  photo

oldpar <- par() # We can save the old configuration using empty par()
par(mfrow=c(1,2))


# Take an image from imageR package - Give me parrots!
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


### Now with something completely different: Local stock image of a laptop
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


### Is the network aware of dinosaurs?
# Load new image
im3 <- load.image("images/dinosaur.jpg")
plot(im3)

# Normalize the laptop
preproc <- preproc.image(im3, mean.img)
plot(as.cimg(preproc[,,,1]))
# The head gets cut, does it matter?
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# African Chamaleon. Almost!


### Agbar
# Load new image
im4 <- load.image("images/agbar.jpg")
plot(im4)

# Normalize the laptop
preproc <- preproc.image(im4, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

# Which class is the most representative
printClassRank(prob, synsets)
# Mitten... Almost. The network is not quite sure about what Agbar tower is.

# Let's try with another photo
im5 <- load.image("images/Agbar2.png")
plot(im5)

# Normalize the laptop
preproc <- preproc.image(im5, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

printClassRank(prob, synsets)
# Waterbottle. Closer.


### You may know about this one. 
im6 <- load.image("images/NES.png")
dim(im6)
# Notice that this image has 4 channels!
plot(im6)

# Remove Alpha Channel
im6 <- rm.alpha(im6)

# Normalize the laptop
preproc <- preproc.image(im6, mean.img)
plot(as.cimg(preproc[,,,1]))
prob <- predict(model, X=preproc)

printClassRank(prob, synsets)
# Modem, Tape player... It is recognizing the aesthetics of that time.



