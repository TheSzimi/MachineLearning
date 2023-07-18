import numpy as np
from scipy.stats import ttest_rel

#files = ['results1.npy', 'results2.npy', 'results3.npy']
#files = ['2results1.npy', '2results2.npy', '2results3.npy']
files = ['3results1.npy', '3results2.npy', '3results3.npy']

#clfs = ['SEA(Oversampling)','SEA(Undersampling)']
#clfs = ['GNB', 'kNN', 'CART']
clfs = ['MySEA', 'SEA', 'AWE']

n_clfs = len(clfs)

for f in files:
    scores = np.load(f)
    #print(scores.shape)

    t_stats = np.zeros((n_clfs, n_clfs))
    p_values = np.zeros((n_clfs, n_clfs))
    advantages = np.zeros((n_clfs, n_clfs), dtype=bool)
    significances = np.zeros((n_clfs, n_clfs), dtype=bool)
    final = np.zeros((n_clfs, n_clfs), dtype=bool)

    for i in range(n_clfs):
        for j in range(n_clfs):
            if i == j:
                continue
            sel_scor_i = scores[:, i, :].flatten()
            sel_scor_j = scores[:, j, :].flatten()
            t_stat, p_value = ttest_rel(sel_scor_i, sel_scor_j)
            t_stats[i, j] = t_stat
            p_values[i, j] = p_value
            if np.mean(sel_scor_i) > np.mean(sel_scor_j):
                advantages[i, j] = True
            if p_value < 0.05:
                significances[i, j] = True

    final = advantages * significances

    print(f"Statystyki t:\n {t_stats} \n")
    print(f"Wartości p:\n {p_values} \n")
    print(f"Przewaga:\n {advantages} \n")
    print(f"Istotność statystyczna:\n {significances} \n")
    print(f"Przewaga istotnie statystyczna:\n {final} \n")

    for i in range(n_clfs):
        for j in range(n_clfs):
            if i != j and final[i, j]:
                clf_i_name = clfs[i]
                clf_j_name = clfs[j]
                score_i = np.mean(scores[:, i, :])
                score_j = np.mean(scores[:, j, :])
                std_i = np.std(scores[:, i, :])
                std_j = np.std(scores[:, j, :])
                print(f"{clf_i_name} with mean = {round(score_i,3)} and std = {round(std_i,3)} better than {clf_j_name} with mean = {round(score_j,3)} and std = {round(std_j,3)}")
    print('\n')
