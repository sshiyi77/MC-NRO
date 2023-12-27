# Neighborhood Repartition-based Oversampling

Python code associated with the paper Neighborhood Repartition-based Oversampling Algorithm for Multiclass Imbalanced Data with Label Noise.

# Usage

Tested on Python 2.7.9.

Because of a high computational cost of the NRO method, code was written to take advantage of a parallel processing, running specific trials of the experiment simultaneously. Because of that, some additonal steps are requried to run the experiment.


To initialize the databases:

```python databases.py```


To start neighborhood repartition-based oversampling:

```python resample.py```


The experimental parameters are added to the 'resample.py' for better testing.