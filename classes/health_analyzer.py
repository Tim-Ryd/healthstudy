import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.stats import *
from analysis.plots import *
from analysis.simulations import *
from analysis.models import *

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

    def plot_bp_vs_age(self, color_by='smoker'):
        """Wrapper för plot_bp_vs_age i plots.py"""
        plot_bp_vs_age(self.df, color_by=color_by)

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
    


    def regression_feature(self, feature):
        """
        Enkel regression för valfri feature mot systolic_bp
        """
        return regression_single_feature(self.df, feature)

    
    def plot_regression_with_residuals(self, feature):
        """
        Enkel regression: target ~ feature
        Plottar både scatter + regressionslinje och residualer.
        """
        reg_res = self.regression_feature(feature)
        preds = reg_res['predictions']

        # Scatter + regressionslinje
        plt.figure(figsize=(6,4))
        plt.scatter(self.df[feature], self.df['systolic_bp'], alpha=0.5)
        plt.plot(np.sort(self.df[feature]), preds[np.argsort(self.df[feature])], color='r')
        plt.xlabel(feature)
        plt.ylabel('Systoliskt blodtryck')
        plt.title(f'Regression: BP vs {feature}')
        plt.show()

        # Residualplot
        residuals = self.df['systolic_bp'] - preds
        plt.figure(figsize=(6,4))
        plt.scatter(self.df[feature], residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(feature)
        plt.ylabel('Residualer')
        plt.title(f'Residualplot: BP ~ {feature}')
        plt.show()

    def pca(self, columns, n_components=2):
        """Wrapper för perform_pca i models.py"""
        return perform_pca(self.df, columns, n_components)
    

    def plot_pca(self, columns, group_col, group_colors, title):
        """Scatterplot av PCA med färgkodning."""
        pca_res = self.pca(columns)
        plot_pca_scatter(pca_res['components'], self.df, group_col, group_colors, title)

    def regression_multiple(self, features, target='systolic_bp'):
        """
        Multipel linjär regression för valda features mot target.
        
        Parameters
        ----------
        features : list of str
            Lista på kolumner som ska användas som prediktorer.
        target : str
            Beroende variabel (default 'systolic_bp').

        Returns
        -------
        dict
            Innehåller model, coefficients, intercept och prediktioner.
        """
        return regression_multiple_features(self.df, features, target)   
    
    def plot_regression_multiple(self, features, target='systolic_bp'):
        """
        Wrapper för multipel regression plot.
        """
        reg_res = self.regression_multiple(features, target)
        plot_regression_multiple(reg_res['predictions'], self.df[target])