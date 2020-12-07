---
layout: post
title: Evaluating Metrics for Machine Learning models
---

## Overview:
Evaluation metrics are used to measure the quality of ML models. While there are many different types of evaluation metrics, it is important to understand what each of them means and how they can be used to make comparisons between models. They are a great tool to help make inferences on our model, enabling us to tune them to improve the overall predictive power of our model.  

## Basics:
Ground truth: Ground truth refers to the real life results of an event. It is the reality you want your model to predict. Practically, your model will never be able to predict the ground truth as ground truth will also have some noise and no model gives a hundred percent accuracy but you want your model to be as close as possible.

True Positives(TP):
Model predicts 1 and Ground truth is 1.

False Positives(FP):
Model predicts 1 and Ground truth is 0.

True Negatives(TN):
Model predicts 0 and Ground truth is 0.

False Negatives(FN):
Model predicts 0 and Ground truth is 1.

## Confusion Matrix
Confusion matrix is a K*K dimension matrix/table (K being the total no of classes which the models have to classify in) which summarizes the model's performance compared to the ground truth. 

Let us consider a dataset with three classes of flowers: setosa versicolor virginica. Once our model makes predictions we would like to study the performance of our model. The most intuitive way would be to see how many predictions did it get right for each class and how many did it misclassify to other classes. Confusion matrix (image below) provides a concise way to represent the information. 
![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/confusion_matrix_1.png)
*the code for generating the above confusion matrix can be found [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)*

From the above image we can infer the following information:
* Our model classified 13 images as setosa and all of them were right.
* Our model classified 16 images as versicolor where it got 10 right and misclassified 6 of them as virginica.
* Our model classified 9 virginica as virginica and all of them were right.

This not only helps us identify which classes we might have to further work on but also draws our attention to the classes between which our model might be getting confused.

<!-- The summary provided by confusion matrix can help us understand other evaluation matrices very easily. 
For the rest of this article, we will be using the data provided from the above example of confusion matrix. -->

## Precision or Positive Predictive Value (PPV)
Precision is the ratio between positive predictions the model predicted correctly by the total no of positive predictions made by the model. Mathematically can be written as:

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/precision_1.PNG)

## Sensitivity or Recall  
Sensitivity is the ratio between positive predictions the model predicted correctly by total no of positives in the ground truth. Mathematically can be written as:

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/sensitivity_1.PNG)

## Specificity
Specificity is the ratio between negative predictions the model predicted correctly by total no of negative predictions in the ground truth. Mathematically can be written as:

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/specificity_1.PNG)

## Accuracy
Accuracy is the ratio between the total number of **correct** predictions by the model by the total number of predictions by the model. Mathematically can be written as:

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/accuracy_1.PNG)


## Need for a single metric:
While Confusion Matrix, Precision, Sensitivity, and Specificity provide great insight into our model performance but using a wide verify of metrics can make it difficult to compare the performance of different models. 
Conventionally, Accuracy is one of the most used performance metrics but is not a viable option to evaluate the performance of an ML model. The reason being accuracy doesn't take precision or recall into account. This can be better illustrated using an example.

Think we are testing a cancer classification model. We have a total of 100 studies in which only 8 studies have cancer while the rest are perfectly normal. If a model classifies all 100 patients to not have cancer then its accuracy is 92%, which misleads us to believe that the model is doing well. 

A much more detailed answer as to why accuracy isn't a good metric for model performance evaluation can be found [here](https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models)

This is where F1 score or dice coefficient comes in handy to provide a single numeric value metric to evaluate the performance of a model.

## F1 score or dice coefficient
The definition of F1 score is the Harmonic Mean between precision and recall. Mathemitically can be written as:
![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/F1_score_1.PNG)

The reason it is of significant importance is because 
*F1 score is summaries many metrics like Recall, Precision, True Positive, False Positive, False Negatives into one
*Very small values of recall or precision will result in a lower F1 score. Thus, it helps in balancing between the two.
*If there is a class imbalance case where the positive class has fewer sample points, then F1-score can help balance the metric across positive/negative samples.

<!-- need to double check this -->
## False positive rate or 1-Specificity
False positive rate is essentially the total **error rate** our models have while trying to predict **positive classification**. 


## ROC curve
The Receiver Operating Characteristic curve is a plot of Sensitivity vs 1-Specificity for different threshold points. It summarizes all of the confusion matrices that each of the thresholds produces. It helps us determine the balance between the Sensitivity and 1-Specificity ie at the cost of how many false positives can we get the desired Sensitivity, helping us choose the best threshold for our desired result.

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/ROC_1.PNG)

* In the graph above at point A, Sensitivity and 1-Specificity both are 1. This is to say all the Positive class points are classified correctly and all the Negative class points are classified incorrectly implying it's as good as a  random classifier to classifying everything positively.
* At point B, sensitivity is same as point A but it has a lower 1-Specificity. Meaning the number of incorrectly Negative class points is lower compared to the previous threshold.
* Between points C and D, the Sensitivity at point C is higher than point D for the same 1-Specificity. This means for the same number of 1-Specificity threshold at c gave us better True Positives than D.
* At point E our 1-Specificity becomes the least implying there are no false positives.

<!-- need to double check this -->
From above it is safe to say the best threshold on ROC curve (with maximum Sensitivity and least 1-Specificity) would lie on the coordinate (0, 1) in the cartesian plane.

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/ROC_2.png)

<!-- need to double check this -->
Note: If keeping false Negatives to a minimal is the desired result then we can plot the graph between Sensitivity and precision. Especially useful in imbalanced data.

## Area under ROC (AUC ROC)
AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1). The auc makes it easy to compare one ROC curve to another helping us choose between different models. In the picture below we can see that the red model is performing better than the blue one.

![_config.yml]({{ site.baseurl }}/images/evaluation_metrics/AUC_ROC_1.PNG)
*image due to the courtesy of the video mentioned below*

Note: Highly recommend watching [this video](https://www.youtube.com/watch?v=4jRBRDbJemM) for a more detailed understanding of ROC.


## References:

* The confusion matrix example has been borrowed from the sklear documentation which can be found at [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
* https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
* https://www.youtube.com/watch?v=4jRBRDbJemM