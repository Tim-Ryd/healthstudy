import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def basic_statistics(df, columns):
    """
    Returnerar en DataFrame med medel, median, min, max, std och n
    för utvalda kolumner.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame som innehåller kolumnerna.
    columns : list of str
        Lista med kolumnnamn att beräkna statistik för.

    Returns
    -------
    pandas.DataFrame
        En tabell med statistiken per kolumn.
    """

    rows = []

    for col in columns:
        series = df[col].dropna()
        rows.append({
            'column': col,
            'mean': series.mean(),
            'median': series.median(),
            'min': series.min(),
            'max': series.max(),
            'std': series.std(ddof=1),
            'n': series.count()
        })

    return pd.DataFrame(rows).set_index('column')

def ttest_smokers_bp(df, smoker_col='smoker', bp_col='systolic_bp'):
    """
    Utför ett Welch t-test för att undersöka om rökare har högre blodtryck
    än icke-rökare (ensidigt test).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame som innehåller kolumnerna.
    smoker_col : str
        Kolumnnamn för rökstatus ('Yes'/'No').
    bp_col : str
        Kolumnnamn för blodtrycksmätningen.

    Returns
    -------
    dict
        Dictionary med t-statistik, tvåsidigt p-värde och ensidigt p-värde.
    """

    smokers = df[df[smoker_col] == 'Yes'][bp_col].dropna()
    nonsmokers = df[df[smoker_col] == 'No'][bp_col].dropna()

    # Welch t-test
    t_stat, p_two_sided = stats.ttest_ind(smokers, nonsmokers, equal_var=False)

    # Ensidigt test: rökare > icke-rökare
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    return {
        't_stat': float(t_stat),
        'p_two_sided': float(p_two_sided),
        'p_one_sided': float(p_one_sided)
    }

def ci_normal_mean(series, confidence=0.95):
    """
    Beräknar normalapproximerat konfidensintervall för medelvärdet.

    Parameters
    ----------
    series : array-like
        Data (t.ex. df['systolic_bp'])
    confidence : float
        Konfidensnivå, default 0.95

    Returns
    -------
    tuple (float, float)
        (lower bound, upper bound)
    """
    data = np.array(series.dropna())
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # z-värde
    z = 1.96 

    margin = z * std / np.sqrt(n)
    return mean - margin, mean + margin


def bootstrap_ci(series, n_boot=1000, ci=95, seed=42):
    """
    Beräknar konfidensintervall för medelvärdet med bootstrap.

    Parameters
    ----------
    series : array-like (pandas Series eller numpy array)
        Data att bootstrap-sampla.
    n_boot : int
        Antal bootstrap-samplingar.
    ci : float
        Konfidensnivå (t.ex. 95 för 95% CI)
    seed : int
        Slumptalsfrö för reproducerbarhet.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) för CI.
    """
    np.random.seed(seed)
    series = pd.Series(series).dropna()
    n = len(series)

    boot_means = [series.sample(n, replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)

    return lower, upper

def power_teoretical_ttest(smokers, nonsmokers, effect_values, alpha=0.05, alternative='larger'):
    """
    Beräknar teoretisk power för olika skillnader i medelvärde mellan två grupper
    med TTestIndPower (statsmodels).

    Returns
    -------
    pd.DataFrame med kolumner ['effect', 'power']
    """
    n1, n2 = len(smokers), len(nonsmokers)
    pooled_std = np.sqrt(smokers.std(ddof=1)**2 + nonsmokers.std(ddof=1)**2)/2
    analysis = TTestIndPower()
    
    powers = []
    for effect in effect_values:
        effect_size = effect / pooled_std  # Cohen's d
        power = analysis.solve_power(effect_size=effect_size,
                                     nobs1=n1,
                                     ratio=n2/n1,
                                     alpha=alpha,
                                     alternative=alternative)
        powers.append(power)
    
 
    return pd.DataFrame({'effect': effect_values, 'power': powers})