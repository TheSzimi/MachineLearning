import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import mode
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_X_y
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score

from strlearn.ensembles.base import StreamingEnsemble
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from strlearn.ensembles import SEA, AWE 


class MySEA(StreamingEnsemble):
    def __init__(self, n_classifiers=10, base_estimator=GaussianNB(), metric=balanced_accuracy_score, base_preprocessing=RandomUnderSampler()):
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

"""
clf = [
    MySEA(base_estimator=GaussianNB()),
    SEA(base_estimator=GaussianNB()),
    AWE(base_estimator=GaussianNB()),
]

metrics = [balanced_accuracy_score]
evaluator = TestThenTrain(metrics)

scores1 = []
scores2 = []
scores3 = []
r_state = [3, 17, 51, 111, 177, 293, 343, 555, 731, 912]

# Dryfy nagłe
for i in range(10):
    scores_per_iteration = []
    for j in range(len(clf)):
        stream1 = StreamGenerator(n_chunks=250, chunk_size=200, n_drifts=3, concept_sigmoid_spacing=999, weights=[0.95, 0.05], random_state=r_state[i])
        evaluator.process(stream1, clf)
        scores_per_iteration.append(evaluator.scores[j, :])
    scores1.append(scores_per_iteration)

np.save('3results1', scores1)

# Dryfy gradualne 
for i in range(10):
    scores_per_iteration = []
    for j in range(len(clf)):
        stream2 = StreamGenerator(n_chunks=250, chunk_size=200, n_drifts=3, concept_sigmoid_spacing=5, weights=[0.95, 0.05], random_state=r_state[i])
        evaluator.process(stream2, clf)
        scores_per_iteration.append(evaluator.scores[j, :])
    scores2.append(scores_per_iteration)

np.save('3results2', scores2)


# Dryfy inkrementalne
for i in range(10):
    scores_per_iteration = []
    for j in range(len(clf)):
        stream3 = StreamGenerator(n_chunks=250, chunk_size=200, n_drifts=3, incremental=True, concept_sigmoid_spacing=5, weights=[0.95, 0.05], random_state=r_state[i])
        evaluator.process(stream3, clf)
        scores_per_iteration.append(evaluator.scores[j, :])
    scores3.append(scores_per_iteration)

np.save('3results3', scores3)
"""
scores1 = np.load('3results1.npy')
scores2 = np.load('3results2.npy')
scores3 = np.load('3results3.npy')

scores1 = np.array(scores1)
mean_scores1 = np.mean(scores1, axis=0)
#print(mean_scores1.shape)
plt.subplot(311)
plt.plot(mean_scores1[0,:,])
plt.plot(mean_scores1[1,:,])
plt.plot(mean_scores1[2,:,])
#plt.xlabel('Liczba chunków')
plt.ylabel('Średnia dokładność')
plt.xlim(0, 250)
plt.ylim(0.5, 1)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.title('Stream 1')
plt.legend(['MySEA','SEA','AWE'])

scores2 = np.array(scores2)
mean_scores2 = np.mean(scores2, axis=0)
#print(mean_scores2.shape)
plt.subplot(312)
plt.plot(mean_scores2[0,:,])
plt.plot(mean_scores2[1,:,])
plt.plot(mean_scores2[2,:,])
#plt.xlabel('Liczba chunków')
plt.ylabel('Średnia dokładność')
plt.xlim(0, 250)
plt.ylim(0.5, 1)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.title('Stream 2')
plt.legend(['MySEA','SEA','AWE'])

scores3 = np.array(scores3)
mean_scores3 = np.mean(scores3, axis=0)
#print(mean_scores3.shape)
plt.subplot(313)
plt.plot(mean_scores3[0,:,])
plt.plot(mean_scores3[1,:,])
plt.plot(mean_scores3[2,:,])
plt.xlabel('Liczba chunków')
plt.ylabel('Średnia dokładność')
plt.xlim(0, 250)
plt.ylim(0.5, 1)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.title('Stream 3')
plt.legend(['MySEA','SEA','AWE'])

plt.suptitle('MySEA  vs SEA vs AWE)')
plt.show()

