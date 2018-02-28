
#install_keras(tensorflow="1.4.1-gpu")

library(keras)
use_backend("tensorflow")
#use_backend("theano")

#mnist <- dataset_mnist()
mnist <- dataset_fashion_mnist()


x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)



model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  #optimizer = optimizer_sgd(momentum=0.9, lr=0.001),
  metrics = c('accuracy')
)


set.seed(1234)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 20, batch_size = 500, 
  validation_split = 0.5
)

plot(history)


mmodel <- model %>% multi_gpu_model(gpus = NULL)

mmodel %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)





history <- mmodel %>% fit(
  x_train, y_train, 
  epochs = 50, batch_size = 5000, 
  validation_split = 0.2
)

plot(history)


model %>% evaluate(x_test, y_test)

model %>% predict_classes(x_test)

rm(mnist)

rm(list=ls())

cifar10 <-dataset_cifar10()
b <- dataset_cifar100()

c <- dataset_fashion_mnist()

str(c)

str(cifar10)
