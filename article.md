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

To illustrate the typical workflow and expected outputs of *randomMachines*,
we provide two reproducible experiments (one regression and one classification)
implemented in the script `execute_examples.R`. The script also writes the
numerical summaries to CSV files and saves the figures as PNG files in the
`results/` directory.

Both experiments follow the same evaluation protocol. First, a dataset is
resampled using 10 bootstrap splits created with `rsample::bootstraps()`. For
each split, models are fitted using the bootstrap training set
(`rsample::analysis()`) and evaluated on the corresponding holdout set
(`rsample::assessment()`). Performance is summarized by the mean and standard
deviation across splits. We emphasize that these examples are designed as a
reproducible demonstration of usage (rather than an exhaustive benchmark), and
that absolute performance will vary with preprocessing, tuning budget, and data
subsampling choices.

## Regression task: Bolsa Família data

For regression, we use a subset of 1000 observations from the `bolsafam` data
distributed with the package, where the response variable $y$ represents the
usage rate of the Bolsa Família program across Brazilian municipalities. We
compare *randomMachines*, Random Forest, and a feed-forward neural network. *randomMachines* is trained with
$B = 25$ bootstrap samples and `cost = 1`; Random Forest
is trained with `ntree = 25`; and the neural network uses two hidden layers
(`hidden = c(5, 3)`) with linear output. Predictive performance is assessed with
root mean squared error (RMSE) on the holdout set of each split.

Table \autoref{tab:regression-results} reports the average RMSE and its
variability across the 10 splits. Under this experimental setup,
*randomMachines* obtains a slightly lower mean RMSE than Random Forest, while
the neural network yields a substantially higher RMSE and larger variability.

| Model            | RMSE (mean) | RMSE (sd) |
|:-----------------|------------:|----------:|
| Random Machines  | 0.01565     | 0.00070   |
| Random Forest    | 0.01623     | 0.00075   |
| Neural Network   | 0.03269     | 0.01019   |

Table: Regression results (mean and standard deviation of RMSE across 10
bootstrap holdout resamples). {#tab:regression-results}

![RMSE distribution across bootstrap holdout resamples for the Bolsa Família
regression task. For readability, the figure shows *randomMachines* and Random
Forest; the neural network results are reported in
\autoref{tab:regression-results}.](results/regression_boxplot.png){#fig:rmse-bolsafam}

## Classification task: Ionosphere radar data

For classification, we evaluate *randomMachines* on the `ionosphere` dataset, a
binary classification benchmark with a nonlinear decision boundary. We compare
*randomMachines* and Random Forest across the same 10 bootstrap holdout splits,
using accuracy on the holdout set as the performance metric. The configuration uses $B = 50$ base learners with `cost = 1` and
`prob_model = FALSE` for *randomMachines*, and `ntree = 50` for Random Forest.

Table \autoref{tab:classification-results} summarizes the mean and standard
deviation of the accuracy values across splits. In this run, *randomMachines* attains a
slightly higher mean accuracy than Random Forest, with comparable variability.

| Model           | Accuracy (mean) | Accuracy (sd) |
|:----------------|----------------:|--------------:|
| Random Machines | 0.93813         | 0.02095       |
| Random Forest   | 0.92818         | 0.01738       |

Table: Classification results (mean and standard deviation of accuracy across
10 bootstrap holdout resamples). {#tab:classification-results}

![Classification accuracy across bootstrap holdout resamples for the ionosphere
task comparing *randomMachines* and Random Forest.](results/classification_boxplot.png){#fig:acc-ionosphere}

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
