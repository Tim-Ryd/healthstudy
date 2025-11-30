import numpy as np
from scipy import stats
import pandas as pd

def simulate_disease_rate(df, n_sim=1000, seed=42):
    """
    Simulerar sjukdomsförekomst baserat på empirisk sannolikhet.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset som innehåller kolumnen 'disease'.
    n_sim : int
        Antal simulerade personer.
    seed : int
        Slumptalsfrö för reproducerbarhet.

    Returns
    -------
    tuple (float, float)
        (verklig andel, simulerad andel)
    """
    np.random.seed(seed)

    disease_rate = df["disease"].mean()
    simulated = np.random.binomial(1, disease_rate, size=n_sim)

    return disease_rate, simulated.mean()

def simulate_power(smokers, nonsmokers, effect_values=[2,3,5,8], n_sim=5000, alpha=0.05, seed=42):
    """
    Simulerar power för olika skillnader i medelvärde mellan två grupper.

    Parameters
    ----------
    smokers : array-like
        Originaldata för grupp 1 (t.ex. rökare).
    nonsmokers : array-like
        Originaldata för grupp 2 (t.ex. icke-rökare).
    effect_values : list of float
        Skillnader i medelvärde att testa.
    n_sim : int
        Antal simuleringar per effekt.
    alpha : float
        Signifikansnivå.
    seed : int
        Slumptalsfrö.

    Returns
    -------
    pd.DataFrame
        Effektstorlek och beräknad power.
    """
    np.random.seed(seed)

    n1, n2 = len(smokers), len(nonsmokers)
    mean_s, mean_ns = smokers.mean(), nonsmokers.mean()
    std_s, std_ns = smokers.std(ddof=1), nonsmokers.std(ddof=1)

    power_simulated = []

    for effect in effect_values:
        significant = 0
        for _ in range(n_sim):
            sim_s = np.random.normal(mean_s + effect, std_s, n1)
            sim_ns = np.random.normal(mean_ns, std_ns, n2)
            t_stat, p_val = stats.ttest_ind(sim_s, sim_ns, equal_var=False)
            # En-sidigt test: rökare > icke-rökare
            if p_val/2 < alpha and np.mean(sim_s) > np.mean(sim_ns):
                significant += 1
        power_simulated.append(significant / n_sim)

    return pd.DataFrame({'effect': effect_values, 'power': power_simulated})