### supervised-classification-algorithms

#### ID3 (Decision Tree)
Constructs a decision tree by selecting the attribute with the highest information gain (entropy reduction) to split the data recursively until purity is achieved or stopping criteria are met.

**Entropy of a set \(S\):**

$$
H(S) = -\sum_{c \in C} p(c) \; \log_2 p(c)
$$

**Information gain per attribute \(A\):**

$$
\text{Gain}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \; H(S_v)
$$


The attribute $A$ that maximises $\text{Gain}(S, A)$ is selected.


#### Naive Bayes

Calculates the prior probability of each class and assumes independence between attributes to estimate the conditional probability, predicting the class that maximises the product (using logarithms for greater numerical stability).

**Posterior probability (Bayes’ theorem with independence):**

$$
P(c_k \mid \mathbf{x}) \propto P(c_k) \prod_{i=1}^{n} P(x_i \mid c_k)
$$

**Predicted class (using logarithms for numerical stability):**

$$
\hat{y} = \underset{c_k}{\arg\max} \left[ \log P(c_k) + \sum_{i=1}^{n} \log P(x_i \mid c_k) \right]
$$

where $P(x_i \mid c_k)$ is estimated using frequency distributions (for categorical variables) or a Gaussian distribution (for continuous variables).


Means and variances are calculated for each feature and class.

A small value (+1e-9) is added to the variances to avoid division by zero or extremely small values that could cause numerical instability.

The Gaussian density function (logPDF) never yields a probability of zero, because the Gaussian bell curve is always positive (albeit very small, never zero). This avoids the problem of zero probabilities that occurs in discrete Naive Bayes when a category does not appear in the training data.



### Naive Bayes Prediction (logarithmic form) (LogPDF)

$$
\log \text{PDF}(x \mid \mu, \sigma^2) = -\frac{1}{2} \log(2\pi \sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

Let `bestLogProb` be the best log-probability found, initialised to $-\infty$.

For each class $c_k$ in the model:

1. The unnormalised posterior log-probability is calculated:

   $$
   \text{logProb}(c_k) = \log P(c_k) + \sum_{i=1}^{n} \log P(x_i \mid c_k)
   $$

   where $\log P(x_i \mid c_k)$ is obtained from the logarithmic density function (logPDF) of a normal distribution with mean $\mu_{k,i}$ and variance $\sigma_{k,i}^2$:

   $$
   \log P(x_i \mid c_k) = -\frac{1}{2}\log(2\pi\sigma_{k,i}^2) - \frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}
   $$

2. If $\text{logProb}(c_k) > \text{bestLogProb}$, then it is updated:

   $$
   \text{bestLogProb} = \text{logProb}(c_k), \quad \text{bestClass} = c_k
   $$

Finally, the class with the highest log-probability is returned:

$$
\hat{y} = \text{bestClass}
$$



[The probability density function (“PDF”)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Normal.logpdf.html)
[Multivariate normal log probability distribution function.](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.multivariate_normal.logpdf.html)
