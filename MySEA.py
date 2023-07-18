
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import mode
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_array, check_X_y
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import balanced_accuracy_score

from strlearn.ensembles.base import StreamingEnsemble
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain

class MySEA(StreamingEnsemble):
    """Algorytm SEA (Streaming Ensemble Algorithm) to metoda uczenia maszynowego stosowana w strumieniach danych, gdzie dane napływają w czasie rzeczywistym. 
    Głównym celem tego algorytmu jest tworzenie dynamicznego zespołu klasyfikatorów, który potrafi adaptować się do zmian w danych oraz dokonywać klasyfikacji na bieżąco.
    Algorytm SEA wyróżnia się tym, że usuwa najgorszy klasyfikator z zespołu. Wyróżniającą cechą mojej implementacji jest dodanie preprocessingu. """
    def __init__(self, n_classifiers=10, base_estimator=GaussianNB(), metric=balanced_accuracy_score, base_preprocessing=RandomOverSampler()):
        self.base_preprocessing = base_preprocessing
        self.metric = metric
        self.base_estimator = base_estimator
        self.n_clfs = n_classifiers
        self.clfs = []

    def partial_fit(self, X, y, classes=None):
        X, y = check_X_y(X, y)
        self.classes = np.unique(y) 
        self.X, self.y = X, y

        for _ in range(self.n_clfs):
            X_resample, y_resample = self.base_preprocessing.fit_resample(self.X, self.y)
            clf = clone(self.base_estimator)
            clf.fit(X_resample, y_resample)
            self.clfs.append(clf)

            if len(self.clfs) > self.n_clfs:
                self.remove_worst_clf()

        return self

    def predict(self, X):
        check_array(X)

        votes = []
        for clf in self.clfs:
            predictions = clf.predict(X)
            votes.append(predictions)
        
        votes= np.array(votes)
        result = mode(votes, axis=0, keepdims=False)[0]

        return result
    
    def remove_worst_clf(self):
        scores = []
        for clf in self.clfs:
            pred = clf.predict(self.X)
            score = self.metric(self.y, pred)
            scores.append(score)
        
        worst_clf = np.argmin(scores)
        del self.clfs[worst_clf]
