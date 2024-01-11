#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(shinycssloaders)
library(fresh)
library(imager)
library(ggplot2)
library(keras)
library(tidyverse)
library(caret)
#reticulate::use_python("C:/Users/f205980/AppData/Local/anaconda3/envs/tf_image")

model=load_model_hdf5("modelo_bom2.hdf5")
escala=1
target_size <- c(64, 64)
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

mytheme <- create_theme(
  adminlte_color(
    light_blue = "#FF0000"
  ),
  adminlte_sidebar(
    width = "300px",
    dark_bg = "#000",
    dark_hover_bg = "#FF0000",
    dark_color = "#FFFFFF"
  ),
  output_file = NULL
)

# Define UI for application that draws a histogram
ui <- dashboardPage(
  
  dashboardHeader(title="Identificador de Urubus"
  ),
  dashboardSidebar(use_theme(mytheme),
                   sidebarMenu(menuItem("Identificador",tabName="ident"),
                               menuItem("Sobre",tabName="sobre"))
  ),
  dashboardBody(
    tabItems(
      tabItem("ident",
              column(12,align="center",
                     h1("Identificador de Urubus"),
                     fileInput("imagem",
                               h4("Insira uma imagem no formato JPG/JPEG e diremos se há um urubu."),
                               buttonLabel = "Procurar",
                               placeholder = "Nenhum arquivo selecionado",
                               accept = "image/*"
                     ),
                     imageOutput("imagem"),
                     htmlOutput("resposta")
              )
              
              
      ),
      tabItem("sobre",
              column(12,align="center",
                     h1("Sobre"),
                     h4("Site dedicado a todos os amantes de urubus. Aqui você poderá inserir
                 uma imagem e ele dirá se nela há ou não um urubu."),
                     h4("É bem provável que ele erre, já que apresenta uma acurácia de 73,44%, mas um dia, quem sabe, eu consigo melhorar o algoritmo. Feito com uma rede neural na linguagem R."),
                     h4("As imagens inseridas nesse site não ficam guardadas."),
                     h4("Criado por: IvoSilvestre")
              )
      )
    )
  )
  
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  output$imagem=renderImage({
    req(input$imagem)
    list(src = input$imagem$datapath,
         width=400,
         height=400
    )
  },
  deleteFile = F
  )
  
  pred_test=reactive(
    model %>% 
      predict(image_prep(input$imagem$datapath)) %>% 
      k_argmax()
  )
  
  output$resposta=renderText({
    ifelse(pred_test()==0,
           HTML("<h4>É um urubu.</h4>"),
           HTML("<h4>Não é um urubu.</h4>")
    )
  }
  )
  
}

# Run the application 
shinyApp(ui = ui, server = server)
