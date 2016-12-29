## Machine Learning (Andrew Ng) Notes #4

标签（空格分隔）： Coursera ML DL AI Andrew_Ng Stanford

---

## A. Non-linear hypotheses

1. For non-linear hypotheses(second polynomial), the features would grow at the rate of $n^2$, where n is the original number of features.
2. Higher order polynomial features might result in over-fitting

## B. Nuerons and the Brain

1. Origin was an algorithm that tried to mimic the brain
2. Very widely used in early 80s and early 90s, popularity diminished in late 90s


## C. Model representation

$a_i^{(j)}$ = activation of unit ***i*** in layer *_j_*

$\theta^{(j)}$ = matrix of weights controlling function mapping from layer _*j*_ to layer **j + 1**
I
if network has $S_j$ units in layer j, $S_{j + 1} $ units in layer *j + 1*, then $\theta^{(j) }$ will be in dimension of $S_{j + 1} * (S_j + 1) $ 

Example: layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of Θ(1) is going to be 4×3 where $S_j$=2 and $S_{j+1}=4$, so $s_{j+1}×(S_j+1) = 4×3 $.

## D.  Multi-class classification

**One-vs-all**


