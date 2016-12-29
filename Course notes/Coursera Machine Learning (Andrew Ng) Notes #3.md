## Machine Learning (Andrew Ng) Notes #3

标签（空格分隔）： Coursera ML DL AI Andrew_Ng Stanford

---
## A. Classification
**定义**: 需要从训练数据中自行分类，依然属于**_Supervised learning_**

**_Logistic regression:_**   
- still a classificatoin algorithm despite of the name.

**Want** classifier: \\( 0 \leq h_\theta(x) \leq 1 \\)       

**_Previous hypothesis function_**: \\( h_\theta(x) = \theta^Tx \\)

**_New hypothethis function_**:
$$ h_\theta(x) = g(\theta^Tx) $$
$$ g(z) = \frac{1}{1+e^{-z}} $$
$$ h_\theta(x) = \frac{1}{1+e^{{-{\theta}^T}x}} $$

g(z) is **_Sigmoid_** function or **_logistic_** function in order to put hypothesis function between 0 and 1.

**\\( h_\theta(x) \\) represents the probability that y = 1 on any input x.**

if \\(h_\theta(x) \geq 0.5 \\): predict y = 1, whenever \\( \theta^Tx >= 0 \\), vice versa

if \\(h_\theta(x) \leq 0.5 \\): predict y = 0, whenever \\( \theta^Tx <= 0 \\), vice versa


**_Decision boundary_**: splits \\(\theta^Tx \geq 0 \\) and \\(\theta^Tx \leq 0 \\), defined by the parameters \\(\theta \\) instead of the dataset itself.

## B. Cost Function

**_原始定义:_** 

$$J(\theta) = \frac{1}{m} \sum_{i=1}^m Cost(h_\theta(x^i),y^i) $$

$$Cost(h_\theta(x), y)=
\begin{cases}
-\log(h_\theta(x))& \text{if y = 1}\\
-\log(1-h_\theta(x))& \text{if y = 0}
\end{cases}$$

or 

$$ Cost(h_\theta(x), y) = -y\log(h_\theta(x)) - (1-y) log(1-h_\theta(x)) $$

每个表达式代表了相应区间的confidence level，当y = 1时，Cost function是0 （代表了zero cost)，但如果此时（新预测的）y变成了0，那么，在这个表达式下对应的cost function值接近无穷大（非常离谱的cost）。

反之亦然，当y = 0时，对应的cost function是0(zero cost)，但如果基于此预测的Y，客观结果是0的时候，在这个表达式下对应的cost function值也会趋近无穷大（非常离谱的cost）。

***Note: y = 0 or 1 always***

Therefore: 

$$J(\theta) = -\frac{1}{m}[ \sum_{i=1}^m y^{(i)}\log(h_\theta(x^{(i)}) + (1-y^{(i)})log(1-h_\theta(x^{(i)})) ]$$

If we were to use the cost function definition for linear regression, ie: 
$$Cost(h_\theta(x), y) = \frac{1}{2}(h_\theta x - y)^2 $$
The hypothesis function for logistic regression will make it non-convex with multiple local optima. Therefore we use the new cost function instead.

## C. Gradident decesent

**_定义:_** 最小化cost function


Want $$ \min_\theta J(\theta) $$

Repeat: $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta) $$

which is:
$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m(h_0(x^i) - y^i) * x^i_j $$

Update all \\( \theta_j \\) simultaneously.

The difference between gradient decesent for linear regression vs for logistic regression is that \\( h_\theta(x) \\) is different. 

Can also run feature scaling on logistic regression to improve performance.


## D. Other optimisions

1. Conjugate gradient
2. BFGS
3. L-BFGS

- The other algorithms automatically change \\( \alpha \\) the learning rate
- Often converges faster than gradient descent
- A lot more complex (duh)

## E. Multiclass classification

**_Definition:_** train a logistic regression classifier \\(h_\theta^{(i)}(x) \\) for each i to predict the probablity that y = i, ie compare each set to the aggregation of all rest datasets.

on a new input x, to make a prediction, pick the class i that maximizes \\( \max (i) h_\theta^{(i)}(x) \\)

Essentially the idea is to separate each group of data from the AGGREGATE of the rest of the data.

## F. Overfitting
**Definition:** When there's too many features, the trained hypothesis can fit the existing samples very well, but fail to generalize on the new data.

Options:

1. Reduce number of features
	- manually select which features to keep
	- model selection algorithm
	- can be less desirable because it fails to use all given features
2. regularization
	- keep all features, but reduce the magnitude / value of such features
	- simpler hypothesis and less prone to overfitting

## G. Regularization for linear regression

**Specifically:**

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^i) - y^i)^2 + \lambda \sum_{i = 1}^m{\theta_j}^2$$

Usually in regularization, $\theta_0$ isn't penalized hence the summation starts from one.

$\lambda \sum_{i = 1}^m{\theta_j}^2$ is the **regularizastion parameter** that targets to:

1. Keep the fitting
2. keep the parameters small

When choosing a large value of $\lambda$, it will effectively result in very small $\theta_n$ that is akin to eliminating the $\theta_1x$.... and leaves $\theta_0$ only. Hence it will result in a flat line in its final hypothesis.

**Gradient descent**


**_Repeat_**

{
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i = 1}^m(h_{\theta}(x^i) - y^i)x_0^i $$

$$\theta_j := \theta_j - \alpha [\frac{1}{m} \sum_{i = 1}^m(h_{\theta}(x^i) - y^i)x_j^i + \frac{\lambda}{m}\theta_j] $$

$$ (j = 1, 2, 3..., n) $$
}

$\theta_0$ is calculated separately because in regularization process, $\theta_0$ is NOT included.

Rewriting $\theta_j$ yields:

$$ \theta_j := \theta_j(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i = 1}^m( h_{\theta}(x^i) - y^i)x_{j}^i $$

The first term $(1 - \alpha \frac{\lambda}{m})$ is usually a bit less than **1**

## H. Normal equation for regularization of linear regression

$$ X = \begin{bmatrix} {(x^{(1)})}^{T} \\ .  \\ . \\. \\ {(x^{(m)})}^T \end{bmatrix} \quad \quad y = \begin{bmatrix} y^{(1)} \\.\\.\\. \\ y^{(m)} \end{bmatrix} $$

To get $\underset{\theta}{\min} J(\theta) $ directly, we will get:


$$ \theta = (X^TX + \lambda \begin{bmatrix} 0 & .  & . & .  \\ 0 & 1 & 0 & .  \\ 0 & . &1 & 0 \\ 0 &. & . & 1  \end{bmatrix})^{-1}X^Ty $$

The matrix is of ( n + 1 ) by ( n + 1 ) dimension with zeroes

## G. Regularized logistic regression

**_Repeat_**

{
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i = 1}^m(h_{\theta}(x^i) - y^i)x_0^i $$

$$\theta_j := \theta_j - \alpha [\frac{1}{m} \sum_{i = 1}^m(h_{\theta}(x^i) - y^i)x_j^i + \frac{\lambda}{m}\theta_j] $$

$$ (j = 1, 2, 3..., n) $$
}