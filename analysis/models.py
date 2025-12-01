import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


def regression_single_feature(df, feature, target='systolic_bp'):
    """
    Enkel linjär regression: target ~ feature

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset med feature och target
    feature : str
        Namnet på kolumnen som ska användas som prediktor
    target : str
        Kolumnnamn för beroende variabel (default 'systolic_bp')

    Returns
    -------
    dict
        Innehåller model, coefficient, intercept och prediktioner
    """
    X = df[feature].values.reshape(-1,1)
    y = df[target].values
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    return {
        'model': model,
        'feature': feature,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'predictions': preds
    }


def perform_pca(df, columns, n_components=2):
    """
    PCA på valda kolumner med standardisering.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame med numeriska kolumner.
    columns : list of str
        Kolumner att använda för PCA.
    n_components : int
        Antal huvudkomponenter att beräkna.
        
    Returns
    -------
    dict
        'pca_model' : PCA-objekt
        'components' : DataFrame med PC1, PC2, ...
        'explained_variance_ratio' : array med andel förklarad varians per komponent
    """
    # Ta bort rader med NaN
    data = df[columns].dropna()
    
    # Standardisera variablerna
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # PCA
    pca_model = PCA(n_components=n_components)
    components = pca_model.fit_transform(data_scaled)
    
    components_df = pd.DataFrame(
        components, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return {
        'pca_model': pca_model,
        'components': components_df,
        'explained_variance_ratio': pca_model.explained_variance_ratio_
    }

def regression_multiple_features(df, features, target='systolic_bp'):
    """
    Multipel linjär regression: target ~ flera features

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset med features och target.
    features : list of str
        Lista med kolumnnamn som ska användas som prediktorer.
    target : str
        Kolumnnamn för beroende variabel (default 'systolic_bp').

    Returns
    -------
    dict
        Innehåller model, coefficients, intercept och prediktioner.
    """
    X = df[features].values
    y = df[target].values
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    return {
        'model': model,
        'features': features,
        'coefficients': dict(zip(features, model.coef_)),
        'intercept': model.intercept_,
        'predictions': preds
    }