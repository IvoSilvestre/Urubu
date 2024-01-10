library(keras)
library(tidyverse)
library(caret)
library(imager)
reticulate::use_python("C:/Users/f205980/AppData/Local/anaconda3/envs/tf_image")

semente=2612
set.seed(semente)
arquivos_aves=list.files("C:/Users/f205980/Desktop/Urubu/Geral2/Aves",full.names = T)
arquivos_urubus=list.files("C:/Users/f205980/Desktop/Urubu/Geral2/Urubus",full.names = T)
arquivos_resto=list.files("C:/Users/f205980/Desktop/Urubu/Geral2/ZResto",full.names = T)

amostra=c(sample(1:length(arquivos_aves),0.2*length(arquivos_aves)),
          sample((length(arquivos_aves)+1):
                   (length(arquivos_aves)+length(arquivos_resto)),
                 0.2*length(arquivos_resto)),
          sample((length(arquivos_aves)+length(arquivos_resto)+1):
                   (length(arquivos_aves)+length(arquivos_resto)+length(arquivos_urubus)),
                 0.2*length(arquivos_urubus))
)

dados=data.frame(filename=c(arquivos_aves,arquivos_resto,arquivos_urubus),
                 class=c(rep("Aves",length(arquivos_aves)),
                         rep("ZResto",length(arquivos_resto)),
                         rep("Urubus",length(arquivos_urubus))
                 )
)
dados=dados[c(amostra,
              (1:length(c(arquivos_aves,arquivos_resto,arquivos_urubus)))[!(1:length(c(arquivos_aves,arquivos_resto,arquivos_urubus)))%in%amostra]),]

escala=1
train_data_gen <- image_data_generator(rescale = escala, # Scaling pixel value
                                       #horizontal_flip = T, # Flip image horizontally
                                       #vertical_flip = T, # Flip image vertically 
                                       #rotation_range = 45, # Rotate image from 0 to 45 degrees
                                       #zoom_range = 0.25, # Zoom in or zoom out range
                                       validation_split = 0.2 # 20% data as validation data
)
target_size <- c(64, 64)

# Batch size for training the model
batch_size <-32
train_image_array_gen <-
  flow_images_from_dataframe(dados, # Folder of the data
                             target_size = target_size, # target of the image dimension (64 x 64)
                             color_mode = "rgb", # use RGB color
                             batch_size = batch_size ,
                             seed = semente,  # set random seed
                             subset = "training", # declare that this is for training data
                             generator = train_data_gen,
                             shuffle = T
  )
val_image_array_gen <-
  flow_images_from_dataframe(dados,
                             target_size = target_size,
                             color_mode = "rgb",
                             batch_size = batch_size ,
                             seed = semente,
                             subset = "validation", # declare that this is the validation data
                             generator = train_data_gen,
                             shuffle = T
  )


train_samples <- train_image_array_gen$n

# Number of validation samples
valid_samples <- val_image_array_gen$n

# Number of target classes/categories
output_n <- n_distinct(train_image_array_gen$classes)

tensorflow::tf$random$set_seed(semente)

model <- keras_model_sequential(name = "simple_model") %>% 
  
  # Convolution Layer
  layer_conv_2d(filters = 16,
                kernel_size = c(3,3),
                padding = "same",
                activation = "linear",
                input_shape = c(target_size, 3) 
  ) %>% 
  
  # Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  # Flattening Layer
  layer_flatten() %>% 
  
  # Dense Layer
  layer_dense(units = 16,
              activation = "linear") %>% 
  
  # Output Layer
  layer_dense(units = output_n,
              activation = "sigmoid",
              name = "Output")
model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = "accuracy"
  )

history <- model %>% 
  fit(
    # training data
    train_image_array_gen,
    
    # training epochs
    steps_per_epoch = as.integer(train_samples / batch_size), 
    epochs =15, 
    
    # validation data
    validation_data = val_image_array_gen,
    validation_steps = as.integer(valid_samples / batch_size)
  )

val_data <- data.frame(file_name =  val_image_array_gen$filenames) %>% 
  # mutate(class = str_extract(file_name, "Aves|Resto|Urubus"))
  mutate(class = str_extract(file_name, "ZResto|Urubus"))
head(val_data, 10)

image_prep <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = target_size, 
                      grayscale = F # Set FALSE if image is RGB
    )
    
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x*escala # rescale image pixel
  })
  do.call(abind::abind, c(arrays, list(along = 1)))
}
test_x <- image_prep(val_data$file_name)
pred_test=model %>% 
  predict(test_x) %>% k_argmax()

decode <- function(x){
  case_when(#x == 0 ~ "Aves",
            x == 0 ~ "Urubus",
            x == 1 ~ "ZResto"
  )
}
pred_test <- sapply(pred_test, decode) 

confusionMatrix(as.factor(pred_test), 
                as.factor(val_data$class)
)

model%>%
  save_model_hdf5("modelo_bom.hdf5")

model2=load_model_hdf5("modelo_bom.hdf5")

imagem=load.image("teste_resized.jpg")

model%>%
  predict(image_prep("teste_resized.jpg"))%>%
  k_argmax()
