hcrf_sparse_python
==================

An hidden conditional random field (HCRF) implementation for text applications 
written in Python.

This software was written during the course of my masters project to show how 
a hidden CRF can be used to classify the sentiment of tweets. For the simple
models we tried, the HCRF unfortunately did not improve the classification
significantly beyond that of using logistic regression. More details can be 
found in the report: *Conditional random fields for noisy text normalisation*.

Usage
---------

hcrf.py can be run as a script to train and test models, and to score input
vectors.
> usage: hcrf.py mode datafile paramfile H [lamb]  
>  
> +   mode:      Set the script mode to train, tst, or label.  
>
> +   datafile:  File containing input datapoints.  
>  +             Format: lines consisting of  
>               label 1 [feat1 feat2 ... ] 2  
>               where label is a non-negative integer, 1 is the special start  
>               of datapoint feature, 2 is the special end of datapoint feature,  
>               and feat1, feat2 etc. are integers > 2 representing features  
>               activated at the first, second etc. time steps.  
>
> +  paramfile: File to store/retrieve parameters.  
>
> + H:         Cardinality of hidden units. Must be >= 3.  
>
> +  lamb:      l2 reguralization constant. Only applicable when mode is train.  

Example
-------

Other stuff
-----------



