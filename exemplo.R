require(tidyverse)
require(randomMachines)
require(randomForest)
require(rsample)
# CURE THE PRINCESS
# DADOS DO ARTIGO


# COMPARAR COM RF TUDO DEFAULT B = 500, HOLDOUT REPETIDO, COMPARAR DEFAULTS
# FAZER COM AMOSTRA COMPLETA

dados_bolsafam <- randomMachines::bolsafam %>% as.data.frame()
bsamples <- bootstraps(dados_bolsafam, times = 10)

# Avaliação Random Machines com melhores hiperparâmetros
rm_resultados <- map_dbl(bsamples$splits, function(split) {
  mod <- randomMachines(
    y ~ .,
    train = analysis(split),
    B = 50,
    cost = 10,
    beta = 0.5
  )
  pred <- predict(mod, assessment(split))
  sqrt(mean((assessment(split)$y - pred)^2))
})

# Avaliação Random Forest padrão com ntree = 500
rf_resultados <- map_dbl(bsamples$splits, function(split) {
  mod <- randomForest(
    y ~ .,
    data = analysis(split),
    ntree = 50
  )
  pred <- predict(mod, assessment(split))
  sqrt(mean((assessment(split)$y - pred)^2))
})

require(neuralnet)

nn_resultados <- map_dbl(bsamples$splits, function(split) {
  mod <- neuralnet(
    y ~ . - REGIAO - COD_UF,
    data = analysis(split),
    hidden = c(5, 3),
    linear.output = TRUE
  )
  pred <- predict(mod, assessment(split))
  sqrt(mean((assessment(split)$y - pred)^2))
})

# Comparação final
tibble(
  modelo = c("Random Machines", "Random Forest", "Neural Network"),
  RMSE_medio = c(mean(rm_resultados), mean(rf_resultados), mean(nn_resultados)),
  RMSE_desvio = c(sd(rm_resultados), sd(rf_resultados), sd(nn_resultados))
)

# DATAFRAME para boxplot

resultados_comparacao <- tibble(
  modelo = rep(c("Random Machines", "Random Forest", "Neural Network"), each = length(rm_resultados)),
  RMSE = c(rm_resultados, rf_resultados, nn_resultados)
)
# Boxplot para comparação
ggplot(resultados_comparacao, aes(x = modelo, y = RMSE)) +
  geom_boxplot() +
  labs(title = "Comparação de RMSE entre Random Machines, Random Forest e Neural Network",
       x = "Modelo",
       y = "RMSE") +
  theme_minimal()


require(mlbench)

# Base simulada para classificação

dados_simulados <- mlbench.spirals(n = 1000, sd = 0.075) %>%
  as.data.frame() %>%
  rename(y = classes)

dados_simulados %>% 
  ggplot(aes(x = x.1, y = x.2, color = y)) +
  geom_point() +
  labs(title = "Dados Simulados para Classificação",
       x = "Feature 1",
       y = "Feature 2") +
  theme_minimal()

# Avaliação Random Machines com melhores hiperparâmetros
data("ionosphere")

dados_class <- ionosphere

colMeans(whosale %>% select(-y))

bsamples_simulados <- bootstraps(dados_class, times = 10)

rm_resultados_simulados <- map_dbl(bsamples_simulados$splits, function(split) {
  mod <- randomMachines(
    y ~ .,
    train = analysis(split),
    B = 50,
    cost = 1,
    prob_model = F
  )
  pred <- predict(mod, assessment(split))
  mean(pred == assessment(split)$y)
})

# Avaliação Random Forest padrão com ntree = 500
rf_resultados_simulados <- map_dbl(bsamples_simulados$splits, function(split) {
  mod <- randomForest(
    y ~ .,
    data = analysis(split),
    ntree = 50
  )
  pred <- predict(mod, assessment(split))
  mean(pred == assessment(split)$y)
})

# Comparação final para classificação
tibble(
  modelo = c("Random Machines", "Random Forest"),
  Acuracia_media = c(mean(rm_resultados_simulados), mean(rf_resultados_simulados)),
  Acuracia_desvio = c(sd(rm_resultados_simulados), sd(rf_resultados_simulados))
)

# Dataframe para boxplot de classificação
resultados_comparacao_simulados <- tibble(
  modelo = rep(c("Random Machines", "Random Forest"), each = length(rm_resultados_simulados)),
  Acuracia = c(rm_resultados_simulados, rf_resultados_simulados)
)
# Boxplot para comparação de classificação
ggplot(resultados_comparacao_simulados, aes(x = modelo, y = Acuracia)) +
  geom_boxplot() +
  labs(title = "Comparação de Acurácia entre Random Machines e Random Forest",
       x = "Modelo",
       y = "Acurácia") +
  theme_minimal()
