import matplotlib.pyplot as plt

def plot_histogram(data, bins, title, xlabel, ylabel):

    """
    Ritar histogram från valfri kolumn.
    """

    plt.figure(figsize=(6,4))
    plt.hist(data, bins=bins,color='r', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_weight_by_gender(df):

    """
    Boxplot av vikt per kön.
    """
    plt.figure(figsize=(6,4))

    data_f = df[df['sex'] == 'F']['weight']
    data_m = df[df['sex'] == 'M']['weight']

    plt.boxplot([data_f, data_m], labels=['F', 'M'])
    plt.title("Boxplot över vikt per kön")
    plt.ylabel("Vikt (kg)")
    plt.show()
    
def plot_bar_smoker_or_not(df):
    """
    Stapeldiagram av andel rökare och icke-rökare.
    """


    plt.figure(figsize=(6,4))
    smoker_counts = df['smoker'].value_counts(normalize=True)

    plt.bar(smoker_counts.index, smoker_counts.values, color=['r','g'])
    plt.title("Andel rökare vs icke-rökare")
    plt.ylabel("Andel")
    plt.show()

def plot_bp_vs_age(df, color_by='smoker'):
    """
    Scatterplot av blodtryck vs ålder.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset som innehåller 'age' och 'systolic_bp'.
    color_by : str
        Kolumnnamn att färgkoda punkterna efter (t.ex. 'smoker', 'sex').
    """
    plt.figure(figsize=(6,4))
    categories = df[color_by].unique()
    for cat in categories:
        subset = df[df[color_by]==cat]
        plt.scatter(subset['age'], subset['systolic_bp'], label=str(cat), alpha=0.7)
    plt.xlabel('Ålder')
    plt.ylabel('Systoliskt blodtryck')
    plt.title('Blodtryck vs ålder')
    plt.legend(title = "Rökare")
    plt.show()

def plot_pca_scatter(components, df, group_col, group_colors, title):

    """
    Plottar ett 2D scatter-diagram av de två första huvudkomponenterna från PCA,
    med punkter färgade efter en gruppering.

    Parametrar
    ----------
    components : pandas.DataFrame
        DataFrame som innehåller minst kolumnerna 'PC1' och 'PC2', 
        vilka representerar de två första huvudkomponenterna.
    df : pandas.DataFrame
        Originaldatasetet som innehåller grupperingkolumnen.
    group_col : str
        Namnet på kolumnen i `df` som används för att tilldela färger till punkterna.
    group_colors : dict
        Ordbok som mappar gruppetiketter i `group_col` till färger.
    title : str
        Titel på scatter-diagrammet.
    """
    plt.figure(figsize=(6,4))
    plt.scatter(
        components['PC1'],
        components['PC2'],
        c=df[group_col].map(group_colors),
        alpha=0.7
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()
    
def plot_regression_multiple(predictions, actual, title='Multipel regression: Prediktioner vs Verkligt värde'):
    """
    Plottar prediktioner från multipel regression mot verkliga värden.

    Parameters
    ----------
    predictions : array-like
        Predikterade värden från modellen.
    actual : array-like
        Verkliga värden (target).
    title : str
        Titel på grafen.
    """
    plt.figure(figsize=(6,4))
    plt.scatter(predictions, actual, alpha=0.5)
    plt.plot([min(predictions), max(predictions)],
             [min(predictions), max(predictions)],
             color='r', linestyle='--')
    plt.xlabel('Prediktioner')
    plt.ylabel('Verkligt värde')
    plt.title(title)
    plt.show()