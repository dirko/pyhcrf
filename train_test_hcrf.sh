#!bin/bash
# Trains an HCRF model on train.dat and stores the parameters in param.dat.
# The model is then tested on the training data and a held out testing set.

# Set the cardinality of the hidden units to 3
H=3
# Set the l2 regularization constant to 1.0
L=1.0 

python hcrf.py train train.dat param.dat $H $L test.dat # Train
python hcrf.py tst train.dat param.dat $H               # Test on training data
python hcrf.py tst test.dat param.dat $H                # Test on testing data
python hcrf.py label train.dat param.dat $H
