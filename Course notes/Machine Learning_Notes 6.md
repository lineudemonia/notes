# Machine Learning Case Study (University of Washington) Notes #6

### Image features
---
- Features = local detectors
	- combined to make prediction
	- in reality, features are more low-level

- image features: collections of locally interesting points
	- combined to build classifiers

### Neural network
---
- Use multiple layers of prediction algorithm to improve prediction accuracy
- Beginning levels of layers, ie deep features of a well-trained neural network, can be used to help train with limited sample of other categories sharing similar attributes to the initial test set.
- ie. neural network transfer

### Code
---


```python
import graphlab
```

### Load the CIFAR-10 dataset


```python
image_train = graphlab.SFrame('image_train_data/')
```


```python
image_test = graphlab.SFrame('image_test_data/')
```

### Exploring the image data


```python
graphlab.canvas.set_target('ipynb')
```


```python
image_train.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">deep_features</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image_array</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.242871761322,<br>1.09545373917, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[73.0, 77.0, 58.0, 71.0,<br>68.0, 50.0, 77.0, 69.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.525087952614, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[7.0, 5.0, 8.0, 7.0, 5.0,<br>8.0, 5.0, 4.0, 6.0, 7.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.566015958786, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[169.0, 122.0, 65.0,<br>131.0, 108.0, 75.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">70</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.12979578972, 0.0, 0.0,<br>0.778194487095, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[154.0, 179.0, 152.0,<br>159.0, 183.0, 157.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">90</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.71786928177, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[216.0, 195.0, 180.0,<br>201.0, 178.0, 160.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">97</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.57818555832, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[33.0, 44.0, 27.0, 29.0,<br>44.0, 31.0, 32.0, 45.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">107</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0,<br>0.220677852631, 0.0,  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[97.0, 51.0, 31.0, 104.0,<br>58.0, 38.0, 107.0, 61.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">121</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.23753464222, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[93.0, 96.0, 88.0, 102.0,<br>106.0, 97.0, 117.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">136</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 7.5737862587, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[35.0, 59.0, 53.0, 36.0,<br>56.0, 56.0, 42.0, 62.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">138</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.658935725689, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[205.0, 193.0, 195.0,<br>200.0, 187.0, 193.0, ...</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>




```python
image_train['image'].show()
```



### Train a classifier on the raw image pixels


```python
raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',
                                              features=['image_array'])
```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    



<pre>WARNING: The number of feature dimensions in this problem is very large in comparison with the number of examples. Unless an appropriate regularization value is set, this model may not provide accurate predictions for a validation/test set.</pre>



<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 1906</pre>



<pre>Number of classes           : 4</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 3072</pre>



<pre>Number of coefficients    : 9219</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy | Validation-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| 1         | 6        | 0.000041  | 4.131434     | 0.377230          | 0.414141            |</pre>



<pre>| 2         | 9        | 0.500000  | 6.237093     | 0.435467          | 0.464646            |</pre>



<pre>| 3         | 10       | 0.500000  | 6.990855     | 0.447534          | 0.464646            |</pre>



<pre>| 4         | 11       | 0.500000  | 7.673936     | 0.442288          | 0.474747            |</pre>



<pre>| 5         | 12       | 0.500000  | 8.323956     | 0.446485          | 0.484848            |</pre>



<pre>| 6         | 13       | 0.500000  | 8.984508     | 0.444386          | 0.474747            |</pre>



<pre>| 7         | 14       | 0.500000  | 9.642127     | 0.467996          | 0.515152            |</pre>



<pre>| 8         | 15       | 0.500000  | 10.302204    | 0.497377          | 0.565657            |</pre>



<pre>| 9         | 16       | 0.500000  | 10.991201    | 0.510493          | 0.484848            |</pre>



<pre>| 10        | 17       | 0.500000  | 11.715536    | 0.515740          | 0.515152            |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>TERMINATED: Iteration limit reached.</pre>



<pre>This model may not be optimal. To improve it, consider increasing `max_iterations`.</pre>


### Make a prediction with the simpole model based on raw pixels


```python
image_test[0:3]['image'].show()
```




```python
image_test[0:3]['label']
```




    dtype: str
    Rows: 3
    ['cat', 'automobile', 'cat']




```python
raw_pixel_model.predict(image_test[0:3])
```




    dtype: str
    Rows: 3
    ['bird', 'cat', 'bird']



#### The raw data model makes all label predictions wrong


```python
raw_pixel_model.evaluate(image_test) 
# Only 46% accuracy
```




    {'accuracy': 0.477, 'auc': 0.7172504999999973, 'confusion_matrix': Columns:
     	target_label	str
     	predicted_label	str
     	count	int
     
     Rows: 16
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |  automobile  |       bird      |  103  |
     |     bird     |       bird      |  525  |
     |     bird     |       dog       |  111  |
     |     dog      |       dog       |  344  |
     |     cat      |       dog       |  222  |
     |     dog      |    automobile   |  170  |
     |     bird     |       cat       |  149  |
     |     cat      |       cat       |  329  |
     |     cat      |    automobile   |  235  |
     |     dog      |       bird      |  258  |
     +--------------+-----------------+-------+
     [16 rows x 3 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns., 'f1_score': 0.4661335125238337, 'log_loss': 1.2202041710835216, 'precision': 0.4682457576391857, 'recall': 0.477, 'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     	class	int
     
     Rows: 400004
     
     Data:
     +-----------+-----+-----+------+------+-------+
     | threshold | fpr | tpr |  p   |  n   | class |
     +-----------+-----+-----+------+------+-------+
     |    0.0    | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   1e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   2e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   3e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   4e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   5e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   6e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   7e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   8e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     |   9e-05   | 1.0 | 1.0 | 1000 | 3000 |   0   |
     +-----------+-----+-----+------+------+-------+
     [400004 rows x 6 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}



### Improve model using deep features (transfer learning)


```python
len(image_train)
```




    2005




```python
#deep_learning_model = graphlab.load_model('http://s3.amazonaws.com/GraphLab-Datasets/deeplearning/imagenet_model_iter45')
#image_train['deep_features'] = deep_learning_model.extract_features(image_train)
```


```python
image_train.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">deep_features</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image_array</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">24</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.242871761322,<br>1.09545373917, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[73.0, 77.0, 58.0, 71.0,<br>68.0, 50.0, 77.0, 69.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.525087952614, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[7.0, 5.0, 8.0, 7.0, 5.0,<br>8.0, 5.0, 4.0, 6.0, 7.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">cat</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.566015958786, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[169.0, 122.0, 65.0,<br>131.0, 108.0, 75.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">70</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.12979578972, 0.0, 0.0,<br>0.778194487095, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[154.0, 179.0, 152.0,<br>159.0, 183.0, 157.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">90</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.71786928177, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[216.0, 195.0, 180.0,<br>201.0, 178.0, 160.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">97</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.57818555832, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[33.0, 44.0, 27.0, 29.0,<br>44.0, 31.0, 32.0, 45.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">107</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">dog</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0,<br>0.220677852631, 0.0,  ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[97.0, 51.0, 31.0, 104.0,<br>58.0, 38.0, 107.0, 61.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">121</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.23753464222, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[93.0, 96.0, 88.0, 102.0,<br>106.0, 97.0, 117.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">136</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 7.5737862587, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[35.0, 59.0, 53.0, 36.0,<br>56.0, 56.0, 42.0, 62.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">138</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">bird</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.658935725689, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[205.0, 193.0, 195.0,<br>200.0, 187.0, 193.0, ...</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>




```python
deep_features_model = graphlab.logistic_classifier.create(image_train,
                                                         features=['deep_features'],
                                                         target='label')
```

    PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.
              You can set ``validation_set=None`` to disable validation tracking.
    



<pre>WARNING: The number of feature dimensions in this problem is very large in comparison with the number of examples. Unless an appropriate regularization value is set, this model may not provide accurate predictions for a validation/test set.</pre>



<pre>WARNING: Detected extremely low variance for feature(s) 'deep_features' because all entries are nearly the same.
Proceeding with model training using all features. If the model does not provide results of adequate quality, exclude the above mentioned feature(s) from the input dataset.</pre>



<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 1911</pre>



<pre>Number of classes           : 4</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 4096</pre>



<pre>Number of coefficients    : 12291</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy | Validation-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>| 1         | 5        | 0.000131  | 4.285963     | 0.736787          | 0.702128            |</pre>



<pre>| 2         | 9        | 0.250000  | 8.336608     | 0.761905          | 0.691489            |</pre>



<pre>| 3         | 10       | 0.250000  | 9.372926     | 0.767138          | 0.691489            |</pre>



<pre>| 4         | 11       | 0.250000  | 10.372975    | 0.775510          | 0.691489            |</pre>



<pre>| 5         | 12       | 0.250000  | 11.384228    | 0.780220          | 0.702128            |</pre>



<pre>| 6         | 13       | 0.250000  | 12.352462    | 0.787023          | 0.712766            |</pre>



<pre>| 7         | 14       | 0.250000  | 13.335912    | 0.801151          | 0.723404            |</pre>



<pre>| 8         | 15       | 0.250000  | 14.333781    | 0.813710          | 0.712766            |</pre>



<pre>| 9         | 16       | 0.250000  | 15.283619    | 0.847200          | 0.755319            |</pre>



<pre>| 10        | 17       | 0.250000  | 16.273401    | 0.855573          | 0.755319            |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+---------------------+</pre>



<pre>TERMINATED: Iteration limit reached.</pre>



<pre>This model may not be optimal. To improve it, consider increasing `max_iterations`.</pre>



```python
image_test[0:3]['label']
```




    dtype: str
    Rows: 3
    ['cat', 'automobile', 'cat']




```python
deep_features_model.predict(image_test[0:3])
```




    dtype: str
    Rows: 3
    ['cat', 'automobile', 'cat']




```python
deep_features_model.evaluate(image_test) 
# Accuracy = 78%
```




    {'accuracy': 0.78325, 'auc': 0.9368902500000005, 'confusion_matrix': Columns:
     	target_label	str
     	predicted_label	str
     	count	int
     
     Rows: 16
     
     Data:
     +--------------+-----------------+-------+
     | target_label | predicted_label | count |
     +--------------+-----------------+-------+
     |  automobile  |       cat       |   11  |
     |     dog      |       cat       |  207  |
     |  automobile  |       dog       |   5   |
     |     cat      |       bird      |   94  |
     |     bird     |       dog       |   53  |
     |     dog      |       bird      |   54  |
     |     cat      |    automobile   |   44  |
     |     bird     |       cat       |  105  |
     |     dog      |    automobile   |   22  |
     |     dog      |       dog       |  717  |
     +--------------+-----------------+-------+
     [16 rows x 3 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns., 'f1_score': 0.7817950543973559, 'log_loss': 0.5744960858632815, 'precision': 0.780726273999268, 'recall': 0.7832499999999999, 'roc_curve': Columns:
     	threshold	float
     	fpr	float
     	tpr	float
     	p	int
     	n	int
     	class	int
     
     Rows: 400004
     
     Data:
     +-----------+----------------+-----+------+------+-------+
     | threshold |      fpr       | tpr |  p   |  n   | class |
     +-----------+----------------+-----+------+------+-------+
     |    0.0    |      1.0       | 1.0 | 1000 | 3000 |   0   |
     |   1e-05   | 0.979333333333 | 1.0 | 1000 | 3000 |   0   |
     |   2e-05   | 0.976666666667 | 1.0 | 1000 | 3000 |   0   |
     |   3e-05   | 0.971666666667 | 1.0 | 1000 | 3000 |   0   |
     |   4e-05   |     0.969      | 1.0 | 1000 | 3000 |   0   |
     |   5e-05   |     0.966      | 1.0 | 1000 | 3000 |   0   |
     |   6e-05   |     0.963      | 1.0 | 1000 | 3000 |   0   |
     |   7e-05   |      0.96      | 1.0 | 1000 | 3000 |   0   |
     |   8e-05   | 0.956666666667 | 1.0 | 1000 | 3000 |   0   |
     |   9e-05   | 0.954333333333 | 1.0 | 1000 | 3000 |   0   |
     +-----------+----------------+-----+------+------+-------+
     [400004 rows x 6 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.}



# Deep features for image retrieval


```python
knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>


**Use image retrieval model with deep features to find similar images**


```python
cat = image_train[18:19]
cat['image'].show()
cat1 = image_test[0:1]
cat1['image'].show()
```






```python
knn_model.query(cat)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 24.794ms     |</pre>



<pre>| Done         |         | 100         | 295.216ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">384</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6910</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9403137951</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39777</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.4634888975</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36870</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.7559623119</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41734</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.7866014148</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>



#### Retrieve image from query results


```python
def get_images_from_ids(query_result):
    return image_train.filter_by(query_result['reference_label'],'id')
```


```python
cat_neighbors = get_images_from_ids(knn_model.query(cat))
cat1_neighbors = get_images_from_ids(knn_model.query(cat1))
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 41.705ms     |</pre>



<pre>| Done         |         | 100         | 273.451ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 15.521ms     |</pre>



<pre>| Done         |         | 100         | 272.74ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
cat_neighbors['image'].show()
cat1_neighbors['image'].show()
```






```python
car = image_train[8:9]
car['image'].show()
```




```python
car_neighbors = get_images_from_ids(knn_model.query(car))
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 26.455ms     |</pre>



<pre>| Done         |         | 100         | 294.585ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
car_neighbors['image'].show()
```



#### Create a lambda functions for image display


```python
show_neighbors = lambda i: get_images_from_ids(knn_model.query(image_train[i:i+1]))['image'].show()
```


```python
show_neighbors(9)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.0498753   | 21.611ms     |</pre>



<pre>| Done         |         | 100         | 270.741ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




### Problem set 6

#### Question 1


```python
image_train['label'].sketch_summary()
```




    
    +------------------+-------+----------+
    |       item       | value | is exact |
    +------------------+-------+----------+
    |      Length      |  2005 |   Yes    |
    | # Missing Values |   0   |   Yes    |
    | # unique values  |   4   |    No    |
    +------------------+-------+----------+
    
    Most frequent items:
    +-------+------------+-----+-----+------+
    | value | automobile | cat | dog | bird |
    +-------+------------+-----+-----+------+
    | count |    509     | 509 | 509 | 478  |
    +-------+------------+-----+-----+------+




#### Least common is bird

#### Question 2


```python
automobile_data = image_train.filter_by('automobile','label') # Filter train data by category
```


```python
cat_data = image_train.filter_by('cat','label')
```


```python
dog_data = image_train.filter_by('dog','label')
```


```python
bird_data = image_train.filter_by('bird','label')
```


```python
automobile_data.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">deep_features</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">image_array</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">97</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.57818555832, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[33.0, 44.0, 27.0, 29.0,<br>44.0, 31.0, 32.0, 45.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">136</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 7.5737862587, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[35.0, 59.0, 53.0, 36.0,<br>56.0, 56.0, 42.0, 62.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">302</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.583938002586, 0.0,<br>0.0, 0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[64.0, 52.0, 37.0, 85.0,<br>60.0, 40.0, 92.0, 66.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">312</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0,<br>0.392823398113, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[124.0, 126.0, 113.0,<br>124.0, 126.0, 113.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">323</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0,<br>4.42310428619, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[241.0, 241.0, 241.0,<br>238.0, 238.0, 238.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">536</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 8.42903900146, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[164.0, 154.0, 154.0,<br>128.0, 119.0, 120.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">593</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[1.65033948421, 0.0, 0.0,<br>0.0, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[231.0, 222.0, 227.0,<br>232.0, 217.0, 221.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">962</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0,<br>0.39552795887, 0.0, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[255.0, 255.0, 255.0,<br>255.0, 255.0, 255.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">997</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.0, 8.04085636139, 0.0, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[145.0, 148.0, 157.0,<br>131.0, 134.0, 145.0, ...</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1421</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Height: 32 Width: 32</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">automobile</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[0.0, 0.0, 0.0, 0.0, 0.0,<br>0.359612941742, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">[114.0, 95.0, 33.0,<br>118.0, 98.0, 26.0, 91.0, ...</td>
    </tr>
</table>
[10 rows x 5 columns]<br/>
</div>




```python
dog_model = graphlab.nearest_neighbors.create(dog_data,features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
cat_model = graphlab.nearest_neighbors.create(cat_data,features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
bird_model = graphlab.nearest_neighbors.create(bird_data,features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



```python
automobile_model = graphlab.nearest_neighbors.create(automobile_data,features=['deep_features'],
                                             label='id')
```


<pre>Starting brute force nearest neighbors model training.</pre>


#### Nearest 'cat' labeled image


```python
get_images_from_ids(cat_model.query(image_test[0:1]))['image'].show()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 25.74ms      |</pre>



<pre>| Done         |         | 100         | 123.253ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




#### Nearest 'dog' labeled image


```python
get_images_from_ids(dog_model.query(image_test[0:1]))['image'].show()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 18.728ms     |</pre>



<pre>| Done         |         | 100         | 107.985ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>




#### Quesiton 3


```python
cat_model.query(image_test[0:1])['distance'].mean()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 24.365ms     |</pre>



<pre>| Done         |         | 100         | 147.441ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





    36.15573070978294



#### Average distance of 5 nearest neighbors in the cat category: 36.15573070978294


```python
dog_model.query(image_test[0:1])['distance'].mean()
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 18.022ms     |</pre>



<pre>| Done         |         | 100         | 97.015ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





    37.77071136184156



#### Average distance of 5 nearest neighbors in the dog category: 37.77071136184156

#### Question 4


```python
image_test_cat = image_test.filter_by('cat','label')
image_test_dog = image_test.filter_by('dog','label')
image_test_bird = image_test.filter_by('bird','label')
image_test_automobile = image_test.filter_by('automobile','label')
```


```python
len(image_test_automobile)
```




    1000




```python
# Understand distance for each trained model using test data from all other categories

dog_cat_neighbors = cat_model.query(image_test_dog, k=1)
bird_cat_neighbors = cat_model.query(image_test_bird, k=1)
automobile_cat_neighbors = cat_model.query(image_test_automobile, k=1)
cat_dog_neighbors = dog_model.query(image_test_cat, k=1)
bird_dog_neighbors = dog_model.query(image_test_bird, k=1)
dog_dog_neighbors = dog_model.query(image_test_dog, k=1)
automobile_dog_neighbors = dog_model.query(image_test_automobile, k=1)
cat_bird_neighbors = bird_model.query(image_test_cat, k=1)
dog_bird_neighbors = bird_model.query(image_test_dog, k=1)
automobile_bird_neighbors = bird_model.query(image_test_automobile, k=1)
cat_automobile_neighbors = automobile_model.query(image_test_cat, k=1)
dog_automobile_neighbors = automobile_model.query(image_test_dog, k=1)
bird_automobile_neighbors = automobile_model.query(image_test_bird, k=1)
```


<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 429.178ms    |</pre>



<pre>| Done         | 509000  | 100         | 503.822ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 128000  | 25.1473     | 492.652ms    |</pre>



<pre>| Done         | 509000  | 100         | 582.283ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 456.897ms    |</pre>



<pre>| Done         | 509000  | 100         | 622.747ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 552.848ms    |</pre>



<pre>| Done         | 509000  | 100         | 652.35ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 474.707ms    |</pre>



<pre>| Done         | 509000  | 100         | 526.044ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 128000  | 25.1473     | 516.399ms    |</pre>



<pre>| Done         | 509000  | 100         | 533.477ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 492.295ms    |</pre>



<pre>| Done         | 509000  | 100         | 571.51ms     |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 120000  | 25.1046     | 430.899ms    |</pre>



<pre>| Done         | 478000  | 100         | 496.167ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 119000  | 24.8954     | 425.816ms    |</pre>



<pre>| Done         | 478000  | 100         | 510.398ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 119000  | 24.8954     | 461.495ms    |</pre>



<pre>| Done         | 478000  | 100         | 519.188ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 464.996ms    |</pre>



<pre>| Done         | 509000  | 100         | 496.658ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 128000  | 25.1473     | 485.137ms    |</pre>



<pre>| Done         | 509000  | 100         | 508.661ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 7668</pre>



<pre>number of reference data blocks: 4</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 127000  | 24.9509     | 438.619ms    |</pre>



<pre>| Done         | 509000  | 100         | 550.884ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
dog_cat_neighbors.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4196077068</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30606</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.8353268874</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5545</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9763410854</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.5750072914</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7493</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.778824791</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47044</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.1171578292</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13918</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.6095830913</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10981</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.9036867306</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45456</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.0674700168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">44673</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.7258732951</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>



#### Create new SFrame with distances from other categories to dogs


```python
dog_distances = graphlab.SFrame({'dog-automobile': dog_automobile_neighbors['distance'],
                                 'dog-bird': dog_bird_neighbors['distance'],
                                 'dog-cat': dog_cat_neighbors['distance'],
                                 'dog-dog': dog_dog_neighbors['distance']})
```


```python
dog_distances.head()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-automobile</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-bird</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-cat</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-dog</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.9579761457</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.7538647304</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.4196077068</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.4773590373</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">46.0021331807</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.3382958925</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.8353268874</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32.8458495684</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.9462290692</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.6157590853</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.9763410854</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.0397073189</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.6866060048</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.0892269954</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.5750072914</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.9010327697</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.2269664935</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.272288694</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.778824791</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.4849250909</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.5845117698</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.1462089236</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.1171578292</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.945165344</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.1067352961</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.523040106</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.6095830913</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.0957278345</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.3221140974</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.1947918393</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.9036867306</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.7696131032</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.8244654995</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.1567131661</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.0674700168</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.1089144603</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.4976929401</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.5597962603</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.7258732951</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.2422832585</td>
    </tr>
</table>
[10 rows x 4 columns]<br/>
</div>




```python
dog_distances[0]['dog-cat']
```




    36.41960770675437



#### If 'dog-dog' = minimum in the current row, return 1. Otherwise return 0


```python
def is_dog_correct(row):
    if row['dog-dog'] == min(row.values()):
        return 1
    else: 
        return 0
```


```python
dog_distances.apply(is_dog_correct).sum()
```




    678



#### 678 out of 1,000 samples in the test data accurately classifies 'dog' in the dog test set.


```python

```

