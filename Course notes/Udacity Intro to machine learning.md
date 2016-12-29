## Intro to machine learning - Udacity

###1. Naive Bayes

~~~ 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(features_train, labesls_train)
pred = clf.prediction(features_test)
accuracy = accuracy_score(labels_test, pred)

print "prediction accuracy is:", accuracy

~~~

###2. Support vector machine

The goal of SVM is to maxmize the marginal distance from different clusters. Marginal distance = minimum of distance from cluster labels

~~~
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)
clf.predict([data])
~~~
**_The kernel trick_**

Map low-dimensional features to high-dimensional spaces that facilitates SVM separation.

Setting kernel = 'rbf' and tuning up C value can drastically improve predictions - overfitting though?

###3. Decision Tree
~~~
from sklearn import tree
X = [[0, 0],[1, 1]]
y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
clf.predict([[2, 2]])
~~~

Entropy: controls how a decision tree decides where to split the data.

Decision tends to **overfit**.

Information gain = entropy(parent) - [weighted average] entropy (children)?


###4. Clustering
Use KMeans to cluster

###5. Text learning

~~~
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
string1, string2, string3
email = [string1, string2, string3]
bag_of_words = vectorizer.fit(email)
bag_of_words = vectorizer.transform(email)
print vectorizer.vocabulary_.get("words) 
# gives the word frequency
~~~

~~~
import nltk
# nltk.download() 
# this can be done once
from nltk.corpus import stopwords
sw = stopwords.words("english")
~~~

####stemming function
~~~
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")
~~~

####tfidf
~~~
tfidf = TfidfVectorizer(stop_words = "english")
X = tfidf.fit_transform(word_data)
feature_names = tfidf.get_feature_names()
print len(feature_names)
~~~

- clean text data
- find stemming root for each word
- run tfidf