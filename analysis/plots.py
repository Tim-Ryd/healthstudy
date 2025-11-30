import matplotlib.pyplot as plt

def plot_histogram(data, bins, title, xlabel, ylabel):
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=bins,color='r', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_weight_by_gender(df):
    plt.figure(figsize=(6,4))

    data_f = df[df['sex'] == 'F']['weight']
    data_m = df[df['sex'] == 'M']['weight']

    plt.boxplot([data_f, data_m], labels=['F', 'M'])
    plt.title("Boxplot över vikt per kön")
    plt.ylabel("Vikt (kg)")
    plt.show()
    
def plot_bar_smoker_or_not(df):
    plt.figure(figsize=(6,4))
    smoker_counts = df['smoker'].value_counts(normalize=True)

    plt.bar(smoker_counts.index, smoker_counts.values, color=['r','g'])
    plt.title("Andel rökare vs icke-rökare")
    plt.ylabel("Andel")
    plt.show()

def plot_scatter_age_bp(df):
    plt.figure(figsize=(6,4))
    plt.scatter(df['age'], df['systolic_bp'])
    plt.title("Blodtryck vs ålder")
    plt.xlabel("Ålder")
    plt.ylabel("Systoliskt blodtryck")
    plt.show()
