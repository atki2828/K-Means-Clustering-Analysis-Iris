## K-Means-Clustering-Analysis-Iris

Build K-Means Clustering Analysis Class and Demonstrate on Fisher's Iris Dataset in Python

## Project Overview

The code in this repo builds an analysis wrapper class for the K-Means Algorithm in SK learn and then demonstrates the methods using the famous Iris data set as an example. By adding 2d and 3d plotting methods to the analysis class we can compare the natural seperation of the data an unsupervised method like K-Means provides vs the true separation in Iris species from the data.

## Files
* KMeans_Class.PY
* Iris_Analysis.PY
* Iris Analysis.ipynb

## Technology Used
* python
* pandas
* numpy
* matplotlib
* sklearn

## Project Takeaways
Both the 3d and 2d comparisons of the natural clusters along with true labels shows that K-means can accurately determine the species of Iris flower given the Sepal Length, Pedal Length and Pedal width. The class wrapper to the sklearn implementation is a good reuseable class to recreate K-means analysis with minimal effort on the part of the analyst.

