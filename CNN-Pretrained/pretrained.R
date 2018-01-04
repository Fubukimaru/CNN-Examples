# From this tutorial: https://mxnet.incubator.apache.org/tutorials/r/classifyRealImageWithPretrainedModel.html

require(mxnet)
require(imager)


model = mx.model.load("Inception/Inception_BN", iteration=39)
mean.img = as.array(mx.nd.load("Inception/mean_224.nd")[["mean_img"]])

# Take an image from imageR package
im <- load.image(system.file("extdata/parrots.png", package="imager"))
plot(im)


# Image preprocess
preproc.image <- function(im, mean.image, dims=3) {
  # Crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  cropped <- crop.borders(im, xx, yy)
  # Resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
  # Convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # Subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

# Normalize the parrots
normed <- preproc.image(im, mean.img)

# Predict the parrots! In other words, get the class probabilities
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Now we get the class names
synsets <- readLines("Inception/synset.txt")


# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))


### LAPTOP
# Load new image
im2 <- load.image("images/laptop.png")
plot(im2)
# Remove Alpha Channel
im2 <- rm.alpha(im2)
plot(im2)
im2

# Normalize the laptop
normed <- preproc.image(im2, mean.img)

# Predict the parrots! In other words, get the class probabilities
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))



### DINOSAUR
# Load new image
im3 <- load.image("images/dinosaur.jpg")
plot(im3)

# Normalize the laptop
normed <- preproc.image(im3, mean.img)
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))
# African Chamaleon. Almost!


### Agbar
# Load new image
im4 <- load.image("images/agbar.jpg")
plot(im4)

# Normalize the laptop
normed <- preproc.image(im4, mean.img)
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))
# Mitten... Almost.


### Agbar
# Load new image
im5 <- load.image("images/Agbar2.png")
plot(im5)

# Normalize the laptop
normed <- preproc.image(im5, mean.img)
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))
# Waterbottle. Closer.


### Agbar
# Load new image
im6 <- load.image("images/NES.png")
dim(im6)
plot(im6)

# Remove Alpha Channel
im6 <- rm.alpha(im6)

# Normalize the laptop
normed <- preproc.image(im6, mean.img)
prob <- predict(model, X=normed)
dim(prob)

# Which class is the most representative
max.idx <- max.col(t(prob))
max.idx

# Get the class name
print(paste0("Predicted Top-class: ", synsets  [[max.idx]]))
# Modem, close enough.

