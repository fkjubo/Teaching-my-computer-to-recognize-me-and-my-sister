# importing libraries

library(keras)
library(EBImage)

# creating a list for training

pic <- c("Dipty1.jpg", "Dipty2.jpg","Dipty3.jpg","Dipty4.jpg",
          "Dipty5.jpg","Dipty6.jpg","Dipty7.jpg","Dipty8.jpg",
          "Dipty9.jpg", "Dipty11.jpg","Dipty12.jpg",
          "Dipty13.jpg","Dipty14.jpg","Dipty15.jpg","Dipty16.jpg",
          "Jubo1.jpg", "Jubo2.jpg","Jubo3.jpg","Jubo4.jpg",
          "Jubo5.jpg","Jubo6.jpg","Jubo7.jpg","Jubo8.jpg",
          "Jubo9.jpg","Jubo10.jpg","Jubo11.jpg","Jubo12.jpg",
          "Jubo13.jpg")

# importing data for training

train <- list()

for (i in 1:28) {train[[i]] <- readImage(pic[i])}

# importing data for testing

picTest <- c("DiptyT1.jpg", "DiptyT2.jpg", "DiptyT3.jpg", "DiptyT4.jpg",
             "DiptyT5.jpg", "DiptyT6.jpg", "DiptyT7.jpg", "DiptyT8.jpg",
             "DiptyT9.jpg", "DiptyT10.jpg", "DiptyT11.jpg", "DiptyT12.jpg",
             "JuboT1.jpg", "JuboT2.jpg", "JuboT3.jpg", "JuboT4.jpg",
             "JuboT5.jpg", "JuboT6.jpg", "JuboT7.jpg", "JuboT8.jpg",
             "JuboT9.jpg", "JuboT10.jpg")

test <- list()

for(i in 1:22) {test[[i]] <- readImage(picTest[i])}

# resizing the dimension of images

for(i in 1:28) {train[[i]] <- resize(train[[i]], 100, 100)}

for(i in 1:22) {test[[i]] <- resize(test[[i]], 100, 100)}

# labeling the labels for train and test data

train.labels <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  1,1,1,1,1,1,1,1,1,1,1,1,1)

test.labels <- c(0,0,0,0,0,0,0,0,0,0,0,0,
                 1,1,1,1,1,1,1,1,1,1)

# creating a labels for testing purpose

test.labels1 <- c(0,0,0,0,0,0,0,0,0,0,0,0,
                 1,1,1,1,1,1,1,1,1,1)


# one hot encoding

train.labels <- to_categorical(train.labels)
test.labels <- to_categorical(test.labels)

# combining all the pictures for the model

train <- combine(train)
test <- combine(test)

# changing the structure of the images for CNN model

train <- aperm(train, c(4,1,2,3))
test <- aperm(test, c(4,1,2,3))

# CNN

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = "relu",
                input_shape = c(100,100,3)) %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = "relu") %>%
  
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  
  layer_flatten() %>%
  
  layer_dense(units = 60, activation = "relu") %>%
  layer_dropout(rate = .35) %>%
  layer_dense(units = 45, activation = "relu") %>%
  layer_dropout(rate = .2) %>%
  layer_dense(units = 30, activation = "relu") %>%
  layer_dropout(rate = .17) %>%
  layer_dense(units = 2, activation = "softmax")

# compile the model

model %>%
  compile(loss = "binary_crossentropy",
          optimizer = "adam",
          metrics = "accuracy")

# fitting data

history <- model %>%
  fit(train,
      train.labels,
      epochs = 120,
      validation_split = .2,
      batch_size = 1)

plot(history)


# predict

model %>% evaluate(test, test.labels)

predictions <- model %>% predict_classes(test)

table(predict = predictions, actual = test.labels1)

print(predictions)