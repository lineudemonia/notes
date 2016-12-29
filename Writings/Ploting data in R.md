##Ploting data in R

**A. What is a good graph for showing two numeric variables?**

- _A scattergraph_
 
```
library(ggplot2)
	
library(dplyr)
	
data(mtcars)
	
ggplot(mtcars, aes(x=drat, y=hp)) + geom_point()
```


**B. What is a good graph for 1 categorical and one numeric variable?**

- _Summarise the numeric variable and make a bar graph_

```
	library(ggplot2)
	library(dplyr)
	data(mtcars)
	mtcars %>% group_by(gear) %>%
	  summarise(hp_median =median(hp)) %>%
	ggplot(aes(x=gear, y=hp_median)) +
	  geom_bar(stat="identity")
```

**C. What is a good graph for two categorical variables?**

- _A bar graph_
 
```
library(ggplot2)
library(dplyr)
data(mtcars)
mtcars$gear <- as.factor(mtcars$gear)
mtcars$cyl <- as.factor(mtcars$cyl)
ggplot(mtcars, aes(x=gear, fill = cyl)) + geom_bar()
```
	  
- _or side by side_

```
ggplot(mtcars, aes(x=gear, fill = cyl)) + geom_bar(position="dodge")
```	  

**D. What is a good graph for two numeric and 1 categorical variable?**

- _A fancy scattergraph._

```
library(ggplot2)
library(dplyr)
data(mtcars)
mtcars$gear <- as.factor(mtcars$gear)
mtcars$cyl <- as.factor(mtcars$cyl)
ggplot(mtcars, aes(x=hp, y=drat, colour=cyl, shape=cyl)) + geom_point(alpha=0.5)
```

- _Or facets_

```
ggplot(mtcars, aes(x=hp, y=drat, colour=cyl)) + geom_point(alpha=0.3) + facet_wrap( ~ cyl, ncol=2) 
```

**E. What is a good graph for three categorical variables?**

- _Bar graph with facets_

```
library(ggplot2)
library(dplyr)
data(mtcars)
mtcars$gear <- as.factor(mtcars$gear)
mtcars$cyl <- as.factor(mtcars$cyl)
mtcars$vs <- as.factor(mtcars$vs)
ggplot(mtcars, aes(x=gear, fill = vs)) + geom_bar(position="dodge") + facet_wrap( ~ cyl, ncol=2) 
```

**F. How to avoid overplotting?**

- _Overplotting is where having points directly on top of each other makes it hard to see how many are there_

```
library(ggplot2)
library(dplyr)
data(mtcars)
ggplot(mtcars, aes(x=drat, y=gear)) +
  geom_point()
```

- _use see through points_

```
ggplot(mtcars, aes(x=drat, y=gear)) +
geom_point(alpha=0.2)
```

- _and/or jittering_

```
ggplot(mtcars, aes(x=drat, y=gear))  + geom_point(alpha=0.4, position=position_jitter(width=.1,height=.1))
```

**G. How to get rid of the grey background?**

- _switch to black and white_

```
library(ggplot2)
library(dplyr)
data(mtcars)
ggplot(mtcars, aes(x=drat, y=gear)) + geom_point() + theme_bw()
```