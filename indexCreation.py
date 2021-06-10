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
from sklearn.model_selection import LeaveOneOut
import scipy
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

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
    pc_values = np.arange(pca.n_components_)+1
    plt.figure(figsize=(10, 8))
    plt.plot(pc_values, pca.explained_variance_, 'bo-', linewidth=2)
    plt.title('Scree Plot of PCA: Component Eigenvalues for '+name)
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained (Eigenvalue)')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.show()

def create_explained_variance_plot(pca, name):
    pcnames = []
    explained_variance = pca.explained_variance_ratio_.tolist()
    print("Explained Variance of PC1: " + str(explained_variance[0]))
    for i in range(len(explained_variance)):
        pcnames.append("PC" + str(i + 1))
    df = pd.DataFrame({'Variance Ratio': explained_variance,
                       'Principal Components': pcnames})
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Principal Components', y="Variance Ratio",
                data=df, color="c").set(title="Variance ratios per component for "+name);
    plt.show()

def cumulative_variance_plot(df_std):
    pca = PCA().fit(df_std)
    PC_values = np.arange(pca.n_components_)+1
    plt.figure(figsize=(10,8))
    plt.plot(PC_values, np.cumsum(pca.explained_variance_ratio_), 'bo--', linewidth=2)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=0.8, linewidth=1, color='r', alpha=0.5)
    plt.show()

def loadings_plot(pca, loadings, columns, name):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})

    # Plot circle
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
    if len(loadings[0]) > 6:
        colors = colors * (int(len(loadings[0]) / 6) + 1)

    add_string = ""
    for i in range(len(loadings[:,0])):
        xi = loadings[i,0]
        yi = loadings[i,1]
        plt.arrow(0, 0,
                  dx=xi, dy=yi,
                  head_width=0.03, head_length=0.03,
                  color=colors[i], length_includes_head=True)
        plt.text(loadings[i,0]*1.2,
                 loadings[i,1]*1.15, fontsize=10,
                 s=columns[i])# + add_string,

    plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)")
    plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)")
    plt.title('Variable factor map (PCA)'+name)
    plt.show()

def score_plot(pca_scores, result_base, name):
    print(pca_scores[:, 0])
    plt.figure(figsize=(11, 8))
    result_base.Region = result_base.Region.astype(int)
    print(pca_scores.shape)
    all_colors = ['red', 'orange', 'gold',
                  'olive', 'limegreen','black', 'seagreen',
                  'turquoise', 'deepskyblue', 'firebrick', 'cyan',
                  'darkblue', 'blue', 'royalblue',
                  'mediumpurple', 'magenta','hotpink', 'pink',
                  'grey', 'sienna', 'orange','yellow','brown']
    color_dict = {}
    for num in range(len(result_base.iloc[:,1].unique())):
        color_dict[result_base.iloc[:,1].unique()[num]] = all_colors[num]

    sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], hue=result_base.iloc[:,1],
                    palette=color_dict, legend='full', s=100).legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Region Number",)
    for i in range(pca_scores.shape[0]):
        plt.text(x=pca_scores[i, 0], y=pca_scores[i, 1], s=result_base.iloc[i,0], fontsize=8)
    plt.xlabel('Principal Component 1 ', fontsize=14)
    plt.ylabel('Principal Component 2 ', fontsize=14)
    plt.axvline(0, ls='--')
    plt.axhline(0, ls='--')
    plt.title("Score plot-")
    for i in range(pca_scores.shape[0]):
        plt.legend([pca_scores[i, 0], pca_scores[i, 1]], result_base.iloc[:,1].map(color_dict))
    plt.legend(title="Region Number", loc="best", fontsize="x-small")
    plt.show()


def biplot(score, loadings, name, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.figure(figsize=(10,8))
    plt.scatter(xs*scalex,ys*scaley, alpha=0.6)
    for i in range(n):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.03, head_length=0.03, length_includes_head=True, color='r', alpha=0.5)
        if labels is None:
            plt.text(loadings[i, 0]* 1.2, loadings[i, 1] * 1.2, "Var" + str(i + 1), color='g',fontsize=7, ha='center', va='center')
        else:
            plt.text(loadings[i, 0]* 1.2, loadings[i, 1] * 1.15, labels[i], color='#306754', fontsize=12,ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(0))
    plt.ylabel("PC{}".format(1))
    plt.title("Biplot - "+name)
    plt.grid()
    plt.show()

def pca_analysis(df, pca_model, isos, pca_scores, name):
    #df_std = StandardScaler().fit_transform(df)
    #pca = PCA()
    #pca_model = pca.fit(df_std)
    #pca_scores = pca.transform(df_std)
    loadings = pca.components_
    #num_pc = pca.n_features_
    print('\nPCA Eigenvalues: \n%s' % pca.explained_variance_)
    loadings_df = get_loadings(pca, df)
    #print('Eigenvectors \n%s' % pca.components_)
    #USING THE KAISER CRITERION, ONLY THE 4 FIRST PRINCIPAL COMPONENTS ARE SIGNIFICANT AND SHOULD BE ANALYZED.
    scree_plot(pca, name)
    create_explained_variance_plot(pca_model, name)
    #cumulative_variance_plot(df_std)
    loadings_plot(pca, loadings, df.columns, name)
    score_plot(pca_scores, name)
    biplot(pca_scores, loadings, name, labels=df.columns.values)

def pairwiseCorr(df):
    # get the column names as list, which are gene names
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])
    result = pd.DataFrame.from_dict(correlations, orient='index')
    result.columns = ['PCC', 'p-value']
    statistically_significant =result.loc[result["p-value"]>0.8]
    highly_correlated = result.loc[abs(result["PCC"])>0.8]
    return statistically_significant.sort_index(), highly_correlated.sort_index()

def get_factor_eigenvalues(df):
    fa = FactorAnalyzer(rotation=None)
    fa.fit(df)
    ev, v = fa.get_eigenvalues()
    return ev

def plot_correlations(df, name):
    plt.figure(figsize=(10, 8))
    plt.title("Correlation plot for "+name)
    sns.heatmap(df.corr(), annot=False, cmap='Blues') #, fmt=".2f"
    plt.xticks(rotation=35)
    plt.show()

def factor_analysis(df, name, not_test_set=True):
    chi_square_value,p_value=calculate_bartlett_sphericity(df)
    kmo_all, kmo_model = calculate_kmo(df)
    p_values, pairwise_correlations = pairwiseCorr(df)
    print("\n"+"Chi_square_value and p_value:")
    print(chi_square_value, p_value)
    print("\n"+"Kmo model:")
    print(kmo_model)
    print("\n"+"Statistically significant correlations:")
    print(p_values)
    print("\n"+"Most correlated variables:")
    print(pairwise_correlations)
    if not_test_set==True:
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

def run_component_analysis(dataset, name, not_test_set):
    print("\n"+"\n"+"\n"+"--------------Investigating features in "+name+" dataset----------------")
    factor_analysis(dataset, name, not_test_set)
    pca_analysis(dataset, name)
    #print("\n"+"\n"+"----Testing with reduced nr of factors----")
    #reduced_correlation_df = remove_correlated_columns(dataset)
    #print(reduced_correlation_df.columns)
    #factor_analysis(reduced_correlation_df, name, all_factors=False)

#Not used
def cross_validation_pca(df):
    train, test = train_test_split(df, shuffle=True, train_size=0.75, test_size=0.25)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    #run_component_analysis(train, "train-world", not_test_set=True)
    scaler = StandardScaler()
    df_std_train = scaler.fit_transform(train)
    df_std_test = scaler.transform(test)
    pca = PCA()
    pca_scores_train = pca.fit_transform(df_std_train)
    pca_scores_test = pca.transform(df_std_test)
    loadings = pca.components_

    scree_plot(pca, "CV")
    print('\nPCA Eigenvalues: \n%s' % pca.explained_variance_)

    cumulative_variance_plot(df_std_train)
    cumulative_variance_plot(df_std_test)

    score_plot(pca_scores_train, "Train")
    score_plot(pca_scores_test, "Test")
    loadings_plot(pca, loadings, df.columns, "Final")
    explained_variance = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_variance)
    print("PCA test with cross validation data - cumulative variance:")
    print(cumsum)

def exp_variance(X, model):
    result = np.zeros(model.n_components_)
    for ii in range(model.n_components_):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)
        result[ii] = 1 - (np.linalg.norm(X_approx_ii - X) /
                          np.linalg.norm(X - model.mean_)) ** 2
    return result.tolist()

def replace_missing_values(df):
    coastal_population = pd.read_excel('External_Data/coastal_population.xls')
    selected_colums = coastal_population[["ISO","2010"]]
    europe_central_asia = selected_colums.loc[selected_colums.ISO == "ECS"]
    latin_america_and_caribbean = selected_colums.loc[selected_colums.ISO == "LAC"]
    south_asia = selected_colums.loc[selected_colums.ISO == "SAS"]
    sub_saharan_africa = selected_colums.loc[selected_colums.ISO == "SSF"]
    east_asia_pacific = selected_colums.loc[selected_colums.ISO == "EAS"]

    for dataset in [europe_central_asia,south_asia,latin_america_and_caribbean,sub_saharan_africa, east_asia_pacific]:
        dataset.reset_index(inplace=True, drop=True)

    for row in range(len(df)):
            if df["ISO"][row] in (["CZE", "HUN", "SVK","KAZ", "KGZ", "ARM", "AUT"]):
                df.loc[row, "Coastal Pop."] = europe_central_asia["2010"][0]
            if df["ISO"][row]=="NPL":
                df.loc[row, "Coastal Pop."] = south_asia["2010"][0]
            if df["ISO"][row]=="PRY":
                df.loc[row, "Coastal Pop."] = latin_america_and_caribbean["2010"][0]
            if df["ISO"][row] in (["RWA", "ZMB"]):
                df.loc[row, "Coastal Pop."] = sub_saharan_africa["2010"][0]
            if df["ISO"][row] =="MNG":
                df.loc[row, "Coastal Pop."] = east_asia_pacific["2010"][0]
    return df

def get_factors_for_index(dataset, cluster=False):
    disaster_interval = dataset.loc[(dataset["Year"]<=2010) & (dataset["Year"]>= 2005)].reset_index(drop=True)
    disaster_situation = disaster_interval.groupby(['ISO']).agg({"Yearly Fatalities":'mean', "Yearly #Disasters":'mean'}).reset_index()
    disaster_situation.rename(columns={'Yearly Fatalities': 'Avg Fatalities (5y)','Yearly #Disasters':'Avg #Disasters (5y)'}, inplace=True)
    relevant_data = dataset.loc[(dataset["Year"] == 2010)].reset_index(drop=True)
    relevant_data = replace_missing_values(relevant_data)
    relevant_data = relevant_data.merge(disaster_situation, on="ISO", how="left")
    vulnerability_data = relevant_data[['ISO','Region','Avg Fatalities (5y)', 'Avg #Disasters (5y)', 'PD',
         'Age Dependency', 'Pop. Growth', 'Urban Pop.', 'Coastal Pop.',
         'GDP','GINI','Per Cap Income', 'Poverty', 'Unemployment',
         'Health Exp.', 'HAQ Index',"Schooling Duration", "Teritary Education"]]
    vulnerability_data= vulnerability_data.dropna().reset_index(drop=True)
    iso_column = vulnerability_data[['ISO', 'Region']]
    pca_data = vulnerability_data.drop(columns={"ISO", 'Region'})
    return pca_data, iso_column, vulnerability_data

def hotelling_squared(data, result_base):
    scaler = StandardScaler()
    data = scaler.fit(data).transform(data)
    pca = PCA()
    pcaFit = pca.fit(data)
    dataProject = pcaFit.transform(data)

    # Calculate ellipse bounds and plot with scores
    theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
    circle = np.array((np.cos(theta), np.sin(theta)))
    sigma = np.cov(np.array((dataProject[:, 0], dataProject[:, 1])))
    ed = np.sqrt(scipy.stats.chi2.ppf(0.99, 2))
    ell = np.transpose(circle).dot(np.linalg.cholesky(sigma) * ed)
    a, b = np.max(ell[:, 0]), np.max(ell[:, 1])  # 95% ellipse bounds
    t = np.linspace(0, 2 * np.pi, 100)
    plt.figure(figsize=(10,8))
    plt.scatter(dataProject[:, 0], dataProject[:, 1])
    plt.plot(a * np.cos(t), b * np.sin(t), color='red')
    for i in range(dataProject.shape[0]):
        plt.text(x=dataProject[i, 0], y=dataProject[i, 1], s=result_base.iloc[i,0], fontsize=7)
    plt.grid(color='lightgray', linestyle='--')
    plt.show()

def remove_outliers(full_data):
    removed_outliers = full_data[~full_data.ISO.isin(["USA", "CHN"])].reset_index(drop=True)
    iso_list = removed_outliers[["ISO", "Region"]]
    removed_outliers = removed_outliers.drop(columns={"ISO", "Region"})
    return removed_outliers, iso_list

def loo_cv(X, result_base):
    cv = LeaveOneOut()
    index_results,index_results2, training_results, testing_results, train_variances2=[],[], [], [], []

    # Enumerate splits
    for train_ix, test_ix in cv.split(X):
        #Split data
        train, test = X.loc[train_ix, :], X.loc[test_ix, :]

        #Scale data
        scaler = StandardScaler()
        df_std_train = scaler.fit_transform(train)
        df_std_test = scaler.transform(test)

        #Fit model
        pca = PCA()
        pca_scores_train = pca.fit_transform(df_std_train)
        pca_scores_test = pca.transform(df_std_test)

        #Evaluate model
        explained_variance = pca.explained_variance_ratio_.tolist()
        train_variance = exp_variance(df_std_train, pca)
        test_variance = exp_variance(df_std_test, pca)
        training_results.append(train_variance[0])
        testing_results.append(test_variance[0])
        index_results.append(explained_variance[0])
    print(np.mean(index_results))
    print(np.std(index_results))
    print(index_results)
    print(training_results)
    print(testing_results)
    index_scores,  df_std = [],0
    return result_base, pca, index_scores, df_std

def train_and_run_model(data, result_base):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(data)

    # Train model
    pca = PCA(random_state=42)
    pca_scores = pca.fit(df_std).transform(df_std)

    #Plot results
    create_explained_variance_plot(pca, "Final run")
    score_plot(pca_scores, result_base, "All Data")
    loadings = pca.components_
    names = data.columns
    loadings_plot(pca, loadings, names, " All Data")

    #Groupings based on variables
    health = loadings[[12,13],:]
    health_names = names[[12,13]]
    education = loadings[[14,15],:]
    education_names = names[[14,15]]
    disaster_exposure = loadings[[0,1],:]
    exposure_names = names[[0,1]]
    population = loadings[[2,3,4,5,6],:]
    population_names = names[[2,3,4,5,6]]
    development = loadings[[7, 8, 9, 10, 11]]
    dev_names = names[[7, 8, 9, 10, 11]]

    natural_groupings=[health, education, disaster_exposure, population,development]
    natural_names = [health_names, education_names, exposure_names, population_names, dev_names]
    for i in range(len(natural_groupings)):
        loadings_plot(pca, natural_groupings[i], natural_names[i], " All Data")
    result_base[["Index"]] = pca_scores[:, 0]
    return result_base, loadings


if __name__ == "__main__":
    df = pd.read_excel("COUNTRY_YEARLY.xlsx")
    worldwide_2010, world_index_df, full_data = get_factors_for_index(df) #66 COUNTRIES WITH NO MISSING DATA RETRIEVED
    #Factor Analysis
    factor_analysis(worldwide_2010, "All Data", not_test_set=True)

    #Investigate and remove outliers
    hotelling_squared(worldwide_2010, world_index_df)
    removed_outliers, removed_outliers_ISO = remove_outliers(full_data)
    #factor_analysis(removed_outliers, "Modified Data", not_test_set=True)

    #Cross validation for empirical performance index
    index_results, pca , index_scores, df_std= loo_cv(removed_outliers, removed_outliers_ISO)

    #Train and run on the final dataset
    vulnerability_index, loadings = train_and_run_model(removed_outliers,removed_outliers_ISO)

    print(vulnerability_index[vulnerability_index.ISO.isin(["BGD","FRA","DEU","VNM","KOR","JPN"])])
