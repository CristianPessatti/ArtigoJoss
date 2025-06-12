---
title: 'randomMachines: A flexible ensemble approach for support vector machines'
tags:
  - R
  - Machine Learning
  - SVM
authors:
  - name: Cristian Pessatti dos Anjos
    equal-contrib: true
    affiliation: 1
  - name: Anderson Ara
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Federal University of Parana, Brazil
   index: 1
date: 30 October 2024
bibliography: paper.bib
editor_options: 
  markdown: 
    wrap: 72
---

# Summary

The forces on stars, galaxies, and dark matter under external
gravitational fields lead to the dynamical evolution of structures in
the universe. The orbits of these bodies are therefore key to
understanding the formation, history, and future state of galaxies. The
field of "galactic dynamics," which aims to model the gravitating
components of galaxies to study their structure and evolution, is now
well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems
require efficient numerical tools, many of which require the same base
code (e.g., for performing numerical orbit integration).

# Statement of need

The *randomMachines* package introduces an ensemble methodology,
employing Support Vector Machines (SVM) as base learners combined with
diverse kernel functions in a bagging structure. This software is
designed to address specific limitations in ensemble modeling,
particularly around flexibility and predictive power in both
classification and regression tasks.

Traditional ensemble techniques, such as Random Forests and standard
bagged SVMs, achieve predictive accuracy by combining multiple base
models. However, these methods face challenges: (1) Random Forests rely
exclusively on decision trees, limiting flexibility in scenarios where
non-linear kernel-based learners might perform better, and (2) standard
SVM-based ensembles often use a single kernel function across all base
learners, which can restrict the model's ability to capture complex
patterns within high-dimensional or nonlinear data. The *randomMachines*
package addresses these gaps by leveraging multiple kernel functions in
the ensemble, enhancing both predictive accuracy and model robustness
across diverse data contexts.

By integrating SVMs with a flexible choice of kernel functions—including
Gaussian, polynomial, Laplacian, and linear kernels—*randomMachines*
introduces a weighted, bagged model that adapts dynamically to data
characteristics. Its implementation builds upon recent research
advancements in ensemble support vector models, specifically those
demonstrating that diverse kernel ensembles can lead to significant
improvements in predictive performance (Ara et al., 2021; Maia et al.,
2021). This approach has shown promise in domains requiring complex
predictive power, such as bioinformatics, image classification, and
financial forecasting, where interactions and non-linearities are
prevalent.

*randomMachines* is thus positioned as an effective tool for researchers
and practitioners seeking enhanced flexibility and performance in
ensemble modeling, expanding the applicability of SVM-based techniques
across a range of scientific and applied disciplines.

## Mathematics

Let $\{(x_i, y_i)\}_{i=1}^{n}$ be a training dataset where $x_i \in \mathbb{R}^p$ represents the 
feature vector and $y_i$ is the target variable, which can be either categorical ($y_i \in \{-1,1\}$ 
for classification) or continuous ($y_i \in \mathbb{R}$ for regression). The *randomMachines* 
method follows a bagging-based approach, incorporating a probabilistic selection of kernel functions 
to enhance model diversity and predictive power.

Given a predefined set of $R$ kernel functions $\{K_r(x, x')\}_{r=1}^{R}$, individual models $h_r(x)$ 
are trained on a validation set. The probability of selecting each kernel is computed differently for 
classification and regression.

For classification, the model selection probability $\lambda_r$ is determined based on accuracy:

$$
\lambda_r = \frac{\log\left(\frac{\text{ACC}_r}{1 - \text{ACC}_r}\right)}{\sum_{i=1}^{R} \log\left(\frac{\text{ACC}_i}{1 - \text{ACC}_i}\right)}
$$

where $\text{ACC}_r$ represents the classification accuracy of model $h_r(x)$.

For regression, the probability of selecting each kernel is determined using the Root Mean Squared Error (RMSE):

$$
\lambda_r = \frac{e^{-\beta \delta_r}}{\sum_{j=1}^{R} e^{-\beta \delta_j}}
$$

where $\delta_r$ is the standardized RMSE of each kernel-based model, and $\beta$ is a regularization 
parameter controlling the penalization of kernels with higher error.

For both classification and regression, $B$ bootstrap samples are drawn from the original training data. 
Each sample is used to train a support vector model $g_b(x)$ using a kernel selected randomly with 
probability $\lambda_r$. The models are assigned weights based on their out-of-bag (OOB) performance.

For classification, the model weight $w_b$ is given by:

$$
w_b = \frac{1}{(1 - \Omega_b)^2}, \quad b = 1, \dots, B
$$

where $\Omega_b$ is the classification error for model $g_b(x)$.

For regression, the weight $w_b$ is defined as:

$$
w_b = \frac{1}{\delta_b^2}, \quad b = 1, \dots, B
$$

where $\delta_b$ is the RMSE of the model $g_b(x)$.

The final ensemble predictions are computed as follows:

For classification is used a weighted majority vote:

$$
G(x) = \text{sgn} \left( \sum_{b=1}^{B} w_b g_b(x) \right)
$$

For regression is used weighted average:

$$
G(x) = \sum_{b=1}^{B} w_b g_b(x)
$$

This methodology ensures that models with lower classification error (or lower RMSE in regression) 
contribute more significantly to the final ensemble decision. The *randomMachines* approach leverages 
the diversity of kernel functions to improve both generalization and robustness across different data distributions.

# Examples

The `randomMachines` package provides a flexible and powerful ensemble modeling framework that 
leverages the diversity of multiple kernel functions within Support Vector Machine (SVM) learners. 
In this section, we provide examples illustrating the use of `randomMachines` in both regression and classification 
settings. We also compare its performance with well-established machine learning techniques, namely Random Forests 
and Neural Networks.

These examples demonstrate the functionality and predictive performance of the package using both real and synthetic 
datasets. All experiments were carried out in R version 4.3 using the packages `randomMachines`, `randomForest`, 
`neuralnet`, and `rsample`.

## Regression Task: Bolsa Família dataset

The first example evaluates the predictive performance of `randomMachines` in a regression context using a subset 
of the *Bolsa Família* dataset, which is included in the package. 
The target variable is the usage rate of the Bolsa Família program across Brazilian municipalities.

We compare the average RMSE across 10 bootstrap holdout resamples for three models:

- `randomMachines` with custom hyperparameters,
- `randomForest` with default parameters and 50 trees,
- a feedforward Neural Network with two hidden layers.

```{r regression-evaluation, message=FALSE, warning=FALSE}
library(tidyverse)
library(randomMachines)
library(randomForest)
library(rsample)
library(neuralnet)

# Sample 1000 observations for evaluation
dados_bolsafam <- bolsafam %>% sample_n(1000) %>% as.data.frame()
bsamples <- bootstraps(dados_bolsafam, times = 10)

# Random Machines
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

# Random Forest
rf_resultados <- map_dbl(bsamples$splits, function(split) {
  mod <- randomForest(y ~ ., data = analysis(split), ntree = 50)
  pred <- predict(mod, assessment(split))
  sqrt(mean((assessment(split)$y - pred)^2))
})

# Neural Network
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

# Summary
tibble(
  model = c("Random Machines", "Random Forest", "Neural Network"),
  RMSE_mean = c(mean(rm_resultados), mean(rf_resultados), mean(nn_resultados)),
  RMSE_sd = c(sd(rm_resultados), sd(rf_resultados), sd(nn_resultados))
)
```

To better visualize the variability and overall distribution of model performance, 
we produce a boxplot comparing RMSE across the bootstrap resamples:

```{r regression-boxplot, message=FALSE, warning=FALSE}
# Boxplot
resultados_comparacao <- tibble(
  model = rep(c("Random Machines", "Random Forest", "Neural Network"), each = length(rm_resultados)),
  RMSE = c(rm_resultados, rf_resultados, nn_resultados)
)

ggplot(resultados_comparacao, aes(x = model, y = RMSE)) +
  geom_boxplot() +
  labs(title = "RMSE Comparison across Bootstrap Resamples",
       x = "Model",
       y = "Root Mean Squared Error (RMSE)") +
  theme_minimal()
```

The results show that `randomMachines` achieves slightly higher RMSE and across 
resamples compared to Random Forest, demonstrating its robustness and competitive 
predictive capability for continuous outcomes.

## Classification Task: Ionosphere radar data

In the second example, we assess `randomMachines` in a binary classification task using the well-known `ionosphere` 
dataset, which contains radar returns labeled as "good" or "bad". 
This dataset is suitable for evaluating non-linear classification algorithms due to its complex decision boundary.

Again, we compare `randomMachines` and `randomForest` across 10 bootstrap holdout resamples, 
computing classification accuracy in each iteration.

```{r classification-evaluation, message=FALSE, warning=FALSE}
data(ionosphere)
bsamples_simulados <- bootstraps(ionosphere, times = 10)

# Random Machines
rm_resultados_simulados <- map_dbl(bsamples_simulados$splits, function(split) {
  mod <- randomMachines(
    y ~ .,
    train = analysis(split),
    B = 50,
    cost = 1,
    prob_model = FALSE
  )
  pred <- predict(mod, assessment(split))
  mean(pred == assessment(split)$y)
})

# Random Forest
rf_resultados_simulados <- map_dbl(bsamples_simulados$splits, function(split) {
  mod <- randomForest(y ~ ., data = analysis(split), ntree = 50)
  pred <- predict(mod, assessment(split))
  mean(pred == assessment(split)$y)
})

# Summary
tibble(
  model = c("Random Machines", "Random Forest"),
  Accuracy_mean = c(mean(rm_resultados_simulados), mean(rf_resultados_simulados)),
  Accuracy_sd = c(sd(rm_resultados_simulados), sd(rf_resultados_simulados))
)
```

We also visualize classification accuracy across bootstrap splits using a boxplot:

```{r classification-boxplot, message=FALSE, warning=FALSE}
# Boxplot
resultados_comparacao_simulados <- tibble(
  model = rep(c("Random Machines", "Random Forest"), each = length(rm_resultados_simulados)),
  Accuracy = c(rm_resultados_simulados, rf_resultados_simulados)
)

ggplot(resultados_comparacao_simulados, aes(x = model, y = Accuracy)) +
  geom_boxplot() +
  labs(title = "Classification Accuracy across Bootstrap Resamples",
       x = "Model",
       y = "Accuracy") +
  theme_minimal()
```

The classification results again highlight the advantages of `randomMachines`. 
It achieves close average accuracy when compared to the standard Random Forest implementation. 
These examples demonstrate how `randomMachines` offers a competitive, flexible, 
and interpretable approach for both regression and classification tasks, especially in complex 
scenarios where leveraging multiple kernels enhances model diversity and predictive strength.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub
without a preferred citation) then you can do it with the example BibTeX
entry below for @fidgit.

For a quick reference, the following citation commands can be used: -
`@author:2001` -\> "Author et al. (2001)" - `[@author:2001]` -\>
"(Author et al., 2001)" - `[@author1:2001; @author2:2001]` -\> "(Author1
et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Caption for example
figure.](figure.png) and referenced from text using
\autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){width="20%"}

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and
Semyeong Oh, and support from Kathryn Johnston during the genesis of
this project.

# References
