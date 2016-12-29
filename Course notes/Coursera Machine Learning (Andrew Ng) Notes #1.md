## Machine Learning (Andrew Ng) Notes #1

标签（空格分隔）： Coursera ML DL AI Andrew_Ng Stanford

---
#### A. Supervised Learning
**定义**: 已知数据有确定对应标签，通过学习现有对应关系预测新对应关系。例如：<u>线性回归</u>

1. Classification: 数据对应<u>**discrete 类别**</u>，如鉴别良性肿瘤或恶性肿瘤
2. Regression: 数据对应<u>**continous 数据**</u>，如通过房子大小预测售价

#### B. Unsupervised Learning
**定义**： 已知数据并无确认标签，需通过分析找出现有数据规律。例如：<u>将新闻按照内容相似性归类 (clustering)</u>

#### C. Class notation:
**m**: number of training examples  
**x**'s: 'input' variable / features  
**y**'s: 'output' variable / 'target' variable  
**(x,y)**: one training example  
**(x^i, y^i)**: the i^th training example



#### D. Sample cost function

**_Hypothesis Function_:** 
$$ h_\theta(x) = \theta_0 + \theta_1x $$

**_Parameters_:**  
$$\theta_0,  \theta_1$$

**_Cost Function_:**

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^i) - y^i)^2 $$

**_Goal_:**  
Minimize: $$J(\theta_0, \theta_1) $$

#### E. Gradient Descent algorithm

**_Definition_**:

The idea is to use a fixed jumping distance * slope ( \\( \frac{\partial}{\partial \theta_j} \\) ) to itereate to the local minimum of \\(\theta_j \\). 

$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j} J(\theta_0, \theta_1) $$

$$ \alpha = learning\_rate $$

Notes: Need to simultaneously update both  \\( \theta_0 \\) and \\( \theta_1 \\)

#### F. Applying gradient descent into cost function

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m(h_0(x^i) - y^i) $$

$$ \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m(h_0(x^i) - y^i) * x^i $$

The cost function can also be directly solved with **_normal equation method_**, however, gradient descent will scale better at large data sets because inverting a matrix has the cost of $ n^3 $. 

When n < 10,000, it makes sense to direct solve for theta with the equation below:

$$ \theta = (X^TX)^{-1} * X^T y  $$

Use smaller \\( \alpha \\) but not too small to keep the rate of convergance.

#### G. Feature scaling
Scaling via range:
$$ \frac{X - \mu}{\underset{max}{X} - \underset{min}{X}} $$