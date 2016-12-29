## Introduction to probability and data

####Baye's theorem:
$$ P(A|B) = \frac{P(\text{A and B)}}{P(B)} $$

####Normal distribution:
1 SD = 68%  
2 SD = 95%  
3 SD = 99.7%

\\( \text{Z score} = \frac{p - \mu}{\sigma} \\)

####Binomial distribution:
if *p* represents probability of success, *(1 chr-p)* represents probability of failure, *n* represents number of independent trials and *k* represents number of successes:

$$ P(\text{k successes in n trials}) = {n \choose k} * p^k (1-p)^{(n-k)} $$

where $$ {n \choose k} = \frac{n!}{k!(n-k)!} $$

**_Expected value_** of binomial distribution: \\( \mu = np \\)

**_Standard deviation_** of binomial distribution: \\( \sqrt{np(1-p)} \\)

**Conditions**:  

- the trial must be independent
- the number of trials, n, must be fixed
- each trial outcome must be classified as a success or failure
- probability P must be same for each trial

#### Using nominal distribution to estimate binomial distribution
- Conditions
	- \\( np \ge 10 \\)
	- \\( n(1-p) \ge 10 \\)
- Approximation
	- Binomial(n,p) \\( \approx \\) Nominal(\\(\mu, \sigma\\)) 
	- where \\( \mu = np \\) and \\( \sigma = \sqrt{np(1-p)} \\)

~~~
#R 
# Given observation, mean and sd, calculate probability
> pnorm(24, mean = 21, sd = 5)
[1] 0.7257469

# Given probability, mean and sd, calculate value
> qnorm(0.9, mean = 1500, sd = 5)
[1] 1506.408

# Binomial distribution:
# Given occurences, sample size, and probability of each occurence, calculate total occurences
> dbinom(8, size = 10, p = 0.13)
[1] 2.77842e-06
> choose(10, 8)
[1] 45

~~~