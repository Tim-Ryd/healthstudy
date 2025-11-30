import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.stats import *
from analysis.plots import *
from analysis.simulations import *

class HealthAnalyzer:
    """Klass som kapslar vanliga analyser för hälsodata.


    Exempel på användning:
    ha = HealthAnalyzer(df)
    ha.describe(['age','weight','height','systolic_bp','cholesterol'])
    ha.plot_smoker_proportions()
    model = ha.regression(['age','weight'],'systolic_bp')


    """


    def __init__(self, df):
        self.df = df.copy()


    # --- Deskriptiv statistik ---
    def describe(self, columns):
        """Returnerar beskrivande statistik för angivna kolumner."""
        return basic_statistics(self.df, columns)
    
    def plot_weight_by_gender(self):
        """Boxplot av vikt per kön"""
        plot_weight_by_gender(self.df)

    def plot_histogram(self,data, bins, title, xlabel, ylabel):
        """Boxplot av rökare vs icke-rökare."""
        plot_histogram(data, bins, title, xlabel, ylabel)

    def plot_bar_smoker_or_not(self):
        "Stapeldiagram av andelen rökare"
        plot_bar_smoker_or_not(self.df)

    def scatter_age_bp(self):
        """Boxplot av rökare vs icke-rökare."""
        plot_scatter_age_bp(self.df)

    def disease_simulation(self, n_sim=1000, seed=42):
        """Kör simulering av sjukdomsförekomst.
        
        Returns
        -------
        tuple (float, float)
            (verklig andel, simulerad andel)
    """
        
        return simulate_disease_rate(self.df, n_sim, seed)
    
    def test_smoker_effect(self):
        """Hypotesprövning: Har rökare högre systoliskt blodtryck än icke-rökare?"""
        return ttest_smokers_bp(self.df)
    
    def ci_systolic_normal(self, confidence=0.95):
        """Normalapproximerat 95% CI för systoliskt blodtryck.
        
        Returns
        -------
        tuple (float, float)
            (lower bound, upper bound)
        """
        
        return ci_normal_mean(self.df['systolic_bp'], confidence)
    
    def ci_systolic_bootstrap(self):
        return bootstrap_ci(self.df['systolic_bp'])
    
    def ci_systolic_comparison(self, bootstrap_n=2000, confidence=0.95, seed=42):
        """
        Beräknar både normalapproximerat och bootstrap-konfidensintervall
        för medelvärdet av systoliskt blodtryck.

        Parameters
        ----------
        bootstrap_n : int
            Antal bootstrap-samplingar.
        confidence : float
            Konfidensnivå, t.ex. 0.95.
        seed : int
            Slumptalsfrö för bootstrap.

        Returns
        -------
        dict
            {'normal_CI': (lower, upper),
             'bootstrap_CI': (lower, upper)}
        """
        bp = self.df['systolic_bp']
        
        normal_ci = ci_normal_mean(bp, confidence=confidence)
        boot_ci = bootstrap_ci(bp, n_boot=bootstrap_n, ci=confidence, seed=seed)
        
        return {
            'normal_CI': normal_ci,
            'bootstrap_CI': boot_ci
        }  
    

    def power_smoker_effect(self, effect_values=[2,3,5,8], n_sim=5000, alpha=0.05, seed=42):
        """Simulerad power-analys för skillnad i medel-blodtryck mellan rökare och icke-rökare."""
        smokers = self.df[self.df['smoker']=='Yes']['systolic_bp'].dropna()
        nonsmokers = self.df[self.df['smoker']=='No']['systolic_bp'].dropna()
        return simulate_power(smokers, nonsmokers, effect_values, n_sim, alpha, seed)
    

    def power_smoker_teoretical(self, effect_values=[2,3,5,8], alpha=0.05):
        smokers = self.df[self.df['smoker']=='Yes']['systolic_bp'].dropna()
        nonsmokers = self.df[self.df['smoker']=='No']['systolic_bp'].dropna()
        return power_teoretical_ttest(smokers, nonsmokers, effect_values, alpha=alpha)