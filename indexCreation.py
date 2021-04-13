import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats.stats import pearsonr
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split

#PCA AND FACTOR ANALYSIS
def createPCAScores(df, name, nr_pca_components):
    df_std = StandardScaler().fit_transform(df)
    pca = PCA(n_components=nr_pca_components, random_state=42)
    pca = pca.fit(df_std)
    pca_scores = pca.transform(df_std)
    return pca_scores, pca

def get_loadings(pca, df):
    loadings = pca.components_
    num_pc = pca.n_features_
    pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
    loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
    loadings_df['variable'] = df.columns.values
    loadings_df = loadings_df.set_index('variable')
    results = loadings_df.transpose()
    print("\nLoadings: ")
    print(results)
    return(results)

def scree_plot(pca, name):
    plt.figure(figsize=(10, 8))
    plt.plot(pca.explained_variance_)
    plt.title('Scree Plot of PCA: Component Eigenvalues for '+name)
    #plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.show()

def create_explained_variance_plot(df_std, name):
    pcnames = []
    pca = PCA().fit(df_std)
    explained_variance = pca.explained_variance_ratio_.tolist()
    for i in range(len(explained_variance)):
        pcnames.append("PC" + str(i + 1))
    df = pd.DataFrame({'Variance Ratio': explained_variance,
                       'Principal Components': pcnames})
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Principal Components', y="Variance Ratio",
                data=df, color="c").set(title="Variance ratios per component for "+name);
    plt.show()

def cumulative_variance_plot(df_std, name):
    pca = PCA().fit(df_std)
    PC_values = np.arange(pca.n_components_)
    plt.figure(figsize=(10,8))
    plt.plot(PC_values, np.cumsum(pca.explained_variance_ratio_), 'bo--', linewidth=2)
    plt.title("Cumulative Explained Variance by Components for "+name)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.8, linewidth=1, color='r', alpha=0.5)
    plt.show()

def loadings_plot(pca, pca_values, columns, name):
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 14})
    # Plot circle
    # Create a list of 500 points with equal spacing between -1 and 1
    x = np.linspace(start=-1, stop=1, num=500)
    # Find y1 and y2 for these points
    y_positive = lambda x: np.sqrt(1 - x ** 2)
    y_negative = lambda x: -np.sqrt(1 - x ** 2)
    plt.plot(x, list(map(y_positive, x)), color='maroon')
    plt.plot(x, list(map(y_negative, x)), color='maroon')
    # Plot smaller circle
    x = np.linspace(start=-0.5, stop=0.5, num=500)
    y_positive = lambda x: np.sqrt(0.5 ** 2 - x ** 2)
    y_negative = lambda x: -np.sqrt(0.5 ** 2 - x ** 2)
    plt.plot(x, list(map(y_positive, x)), color='maroon')
    plt.plot(x, list(map(y_negative, x)), color='maroon')

    # Create broken lines
    x = np.linspace(start=-1, stop=1, num=30)
    plt.scatter(x, [0] * len(x), marker='_', color='maroon')
    plt.scatter([0] * len(x), x, marker='|', color='maroon')

    # Define color list
    colors = ['blue', 'red', 'green', 'black', 'purple', 'brown']
    if len(pca_values[0]) > 6:
        colors = colors * (int(len(pca_values[0]) / 6) + 1)

    add_string = ""
    for i in range(len(pca_values[0])):
        xi = pca_values[0][i]
        yi = pca_values[1][i]
        plt.arrow(0, 0,
                  dx=xi, dy=yi,
                  head_width=0.03, head_length=0.03,
                  color=colors[i], length_includes_head=True)
        #add_string = f" ({round(xi, 2)} {round(yi, 2)})"
        plt.text(pca_values[0, i],
                 pca_values[1, i], fontsize=6,
                 s=columns[i])# + add_string,

    plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)")
    plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)")
    plt.title('Variable factor map (PCA)'+name)
    plt.show()

def score_plot(x_pca, name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x_pca[:, 0], x_pca[:, 1],
                    palette="Set1", legend='full', s=100).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Principal Component 1 ', fontsize=14)
    plt.ylabel('Principal Component 2 ', fontsize=14)
    plt.axvline(0, ls='--')
    plt.axhline(0, ls='--')
    plt.title("Score plot-"+name)
    plt.show()

def biplot(score,coeff,pcax,pcay,name, labels=None):
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.figure(figsize=(10,8))
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.title("Biplot - "+name)
    plt.grid()
    plt.show()

def pca_analysis(df, name):
    if not (name=="test-world"):
        df_std = StandardScaler().fit_transform(df)
        pca = PCA()
        pca_scores = pca.fit(df_std).transform(df_std)
    loadings = pca.components_
    num_pc = pca.n_features_
    print('\nPCA Eigenvalues: \n%s' % pca.explained_variance_)
    loadings_df = get_loadings(pca, df)
    #print('Eigenvectors \n%s' % pca.components_)
    #USING THE KAISER CRITERION, ONLY THE 4 FIRST PRINCIPAL COMPONENTS ARE SIGNIFICANT AND SHOULD BE ANALYZED.
    scree_plot(pca, name)
    create_explained_variance_plot(df_std, name)
    cumulative_variance_plot(df_std, name)
    loadings_plot(pca, loadings, df.columns, name)
    score_plot(pca_scores, name)
    biplot(pca_scores, loadings, 1, 2, name, labels=df.columns.values)

def pairwiseCorr(df):
    # get the column names as list, which are gene names
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])
    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    return result.sort_index()

def get_factor_eigenvalues(df):
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df)
    ev, v = fa.get_eigenvalues()
    #fa4 = FactorAnalyzer(n_factors=4, rotation="varimax")
    #fa4.fit(df)
    #ev4, v4 = fa4.get_eigenvalues()
    return ev

def plot_correlations(df, name):
    plt.figure(figsize=(10, 8))
    plt.title("Correlation plot for "+name)
    sns.heatmap(df.corr(), annot=False) #, fmt=".2f"
    plt.show()

def factor_analysis(df, name, all_factors=True):
    chi_square_value,p_value=calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)
    pairwise_correlations = pairwiseCorr(df)
    print("\n"+"Chi_square_value and p_value:")
    print(chi_square_value, p_value)
    print("\n"+"Kmo model:")
    print(kmo_model)
    print("\n"+"Pairwise correlations:")
    print(pairwise_correlations)
    if all_factors==True:
        eigenvalues = get_factor_eigenvalues(df)
        print("\n" + "FA Eigenvalues:")
        print(eigenvalues)
    plot_correlations(df, name)

def remove_correlated_columns(dataset, threshold=0.8):
    reduced_dataset=dataset.copy()
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = reduced_dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in reduced_dataset.columns:
                    del reduced_dataset[colname] # deleting the column from the dataset
    return reduced_dataset

def describe_data(df_pca):
    summary = df_pca.describe()
    summary = summary.transpose()
    print(summary)

def cross_validation_pca(df):
    train, test = train_test_split(df, shuffle=True, train_size=0.75, test_size=0.25)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    run_component_analysis(train, "train-world")
    factor_analysis(test, "test-world") #Can not do all the pca_analysis_tests on the test set

    base = pd.Series(range(len(train)))
    print("\n")
    print("**Cross validation and PCA training***")
    print("Column names:")
    print(train.columns.values)
    print("Train_Set length: "+str(len(train)))
    print("Test_Set length: "+str(len(test))+"\n")

    #Normalize datasets
    sc = StandardScaler()
    X_train_norm = sc.fit_transform(train)
    X_test_norm = sc.transform(test)

    #Test run PCA
    pca = PCA()
    X_train = pca.fit_transform(X_train_norm)
    X_test = pca.transform(X_test_norm)
    explained_variance = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_variance)
    print("PCA test with cross validation data - cumulative variance:")
    print(cumsum)
    #6 components are needed to represent 80% of the variance

    #Run PCA with 6 components:
    pca = PCA(n_components=6, random_state=123)
    scores_pca =pca.fit_transform(X_train_norm)
    DF_PCA = pd.DataFrame(scores_pca, columns=["PC%d" % k for k in range(1, 6 + 1)]).iloc[:, :]
    describe_data(DF_PCA)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    #Test more here****

def run_component_analysis(dataset, name):
    print("\n"+"\n"+"\n"+"--------------Investigating features in "+name+" dataset----------------")
    #print(dataset.info())
    #print(dataset.describe())
    factor_analysis(dataset, name)
    pca_analysis(dataset, name)

    #print("\n"+"\n"+"----Testing with reduced nr of factors----")
    #reduced_correlation_df = remove_correlated_columns(dataset)
    #print(reduced_correlation_df.columns)
    #factor_analysis(reduced_correlation_df, name, all_factors=False)


def get_factors_for_index(dataset, cluster=False):
    if cluster==True:
        relevant_data = dataset.loc[(dataset.Segment == "Cluster 12") & (dataset.Year == 2010)].reset_index(drop=True)
    else:
        relevant_data = dataset.loc[(dataset.Year == 2010)]
        relevant_data= relevant_data.dropna().reset_index(drop=True)

    iso_column = relevant_data[['ISO']]
    vulnerability_data = relevant_data[['Yearly_Fatality_Sum', 'Yearly_Disaster_Count', 'Population Density',
         'Age Dependency', 'Population Growth', 'Rural Population', 'Urban Population', 'Coastal Population',
         'GDP', 'GINI', 'Per Capita Income', 'Poverty Headcount', 'Unemployment Rate',
         'Health Expenditure', 'HAQ_Index', 'AVG Years of Schooling', 'Population with tertiary schooling']]
    return vulnerability_data, iso_column


if __name__ == "__main__":
    df = pd.read_excel("COUNTRY_YEARLY.xlsx")
    cluster_2010, cluster_index_df = get_factors_for_index(df, cluster=True)
    worldwide_2010, world_index_df = get_factors_for_index(df, cluster=False) #ONLY 48 COUNTRIES WITH NO MISSING DATA RETRIEVED

    #TESTING WITH MAX NR OF COUNTRIES:
    dataset_in_focus = worldwide_2010 #cluster_2010
    name_in_focus = " world" #" cluster countries"
    run_component_analysis(dataset_in_focus,name_in_focus) #change dataset to cluster2010 here to see only cluster
    cross_validation_pca(dataset_in_focus)

    #Add scores to dataframe with ISO
    cluster_index_df[["V1", "V2", "V3", "V4"]], pca = createPCAScores(cluster_2010, "cluster", nr_pca_components=4)
    world_index_df[["V1", "V2", "V3", "V4"]], pca = createPCAScores(worldwide_2010, "world",  nr_pca_components=4)

