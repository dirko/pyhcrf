pyHCRF
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
>                label 1 [feat1 feat2 ... ] 2  
>                where label is a non-negative integer, 1 is the special start  
>                of datapoint feature, 2 is the special end of datapoint feature,  
>                and feat1, feat2 etc. are integers > 2 representing features  
>                activated at the first, second etc. time steps.  
>
> +  paramfile: File to store/retrieve parameters.  
>
> + H:         Cardinality of hidden units. Must be >= 3.  
>
> +  lamb:      l2 reguralization constant. Only applicable when mode is train.  

Example
-------
If we have two examples of messages with positive sentiment:
> 1: This is awesome!  
> 2: cool!

and two examples of negative messages:
> 3: This sucks.  
> 4: would not recommend.

then we have a lexicon composed of the following tokens, along with a numbering:

|  START  |  END  |  !|  .| awesome| cool| is| not| recommend| sucks| this| would | 
|---------|-------|---|---|--------|-----|---|----|----------|------|-----|-------|
|        1|      2|  3|  4|       5|    6|  7|   8|         9|    10|  11 |     12|

We can now encode the first message as a sequence of features that are activated
one after the other:
> 1: 1 11 7 5 2

When we choose the label 0 for positive messages and 1 for negative messages, our
training data, train.dat, looks like:
> 0 1 11 7 5 2  
> 0 1 6 3 2  
> 1 1 11 10 2  
> 1 1 12 8 9 2

A model can be trained then with
```bash
python hcrf.py train train.dat params.dat 5 0.1
```

When a model is tested or labeled, the labels in the input file are still necessary 
but are ignored
```bash
python hcrf.py tst train.dat params.dat 5    # gives accuracy on training data
python hcrf.py label train.dat params.dat 5  # prints out sequence of predicted labels
```

In the example above, the cardinality of the hidden variable is set to `S=5`.
The HCRF state machine starts in state 0 for the **START** feature and ends
in state `S-1` with the **END** feature. In between, the machine is currently
constrained to only allow transitions to higher states:
```
( 0 )---->( 1 )---->( 2 )---->( 3 )---->( 4 )
           | ^       | ^       | ^      
            U         U         U
```


