import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import mode
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_array, check_X_y
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score

from strlearn.ensembles.base import StreamingEnsemble
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain

from MySEA import MySEA

clf = [
    MySEA(base_preprocessing=RandomOverSampler()),
    MySEA(base_preprocessing=RandomUnderSampler()),
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

np.save('results1', scores1)

# Dryfy gradualne 
for i in range(10):
    scores_per_iteration = []
    for j in range(len(clf)):
        stream2 = StreamGenerator(n_chunks=250, chunk_size=200, n_drifts=3, concept_sigmoid_spacing=5, weights=[0.95, 0.05], random_state=r_state[i])
        evaluator.process(stream2, clf)
        scores_per_iteration.append(evaluator.scores[j, :])
    scores2.append(scores_per_iteration)

np.save('results2', scores2)

# Dryfy inkrementalne
for i in range(10):
    scores_per_iteration = []
    for j in range(len(clf)):
        stream3 = StreamGenerator(n_chunks=250, chunk_size=200, n_drifts=3, incremental=True, concept_sigmoid_spacing=5, weights=[0.95, 0.05], random_state=r_state[i])
        evaluator.process(stream3, clf)
        scores_per_iteration.append(evaluator.scores[j, :])
    scores3.append(scores_per_iteration)

np.save('results3', scores3)

"""
scores1 = np.load('results1.npy')
scores2 = np.load('results2.npy')
scores3 = np.load('results3.npy')
"""
scores1 = np.array(scores1)
mean_scores1 = np.mean(scores1, axis=0)
#print(mean_scores1.shape)
plt.subplot(311)
plt.plot(mean_scores1[0,:,])
plt.plot(mean_scores1[1,:,])
plt.xlabel('Liczba chunków')
plt.xlim(0, 250)
plt.ylabel('Średnia dokładność')
plt.ylim(0.5, 1)
plt.title('Stream 1')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.legend(['RandomOverSampler','RandomUnderSampler'])

scores2 = np.array(scores2)
mean_scores2 = np.mean(scores2, axis=0)
#print(mean_scores2.shape)
plt.subplot(312)
plt.plot(mean_scores2[0,:,])
plt.plot(mean_scores2[1,:,])
plt.xlabel('Liczba chunków')
plt.xlim(0, 250)
plt.ylabel('Średnia dokładność')
plt.ylim(0.5, 1)
plt.title('Stream 2')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.legend(['RandomOverSampler','RandomUnderSampler'])

scores3 = np.array(scores3)
mean_scores3 = np.mean(scores3, axis=0)
#print(mean_scores3.shape)
plt.subplot(313)
plt.plot(mean_scores3[0,:,])
plt.plot(mean_scores3[1,:,])
plt.xlabel('Liczba chunków')
plt.xlim(0, 250)
plt.ylabel('Średnia dokładność')
plt.ylim(0.5, 1)
plt.title('Stream 3')
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.legend(['RandomOverSampler','RandomUnderSampler'])

plt.suptitle('SEA (oversampling vs undersampling)')
plt.show()

