from __future__ import print_function, division
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from homework2_rent import score_rent

def test_rent():
	assert score_rent() > 0.25
