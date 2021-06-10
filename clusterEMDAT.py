import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_hex, Normalize, from_levels_and_colors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats.stats import pearsonr
from yellowbrick.cluster import SilhouetteVisualizer
import itertools

#*******************************START PROCESSING **************************************#

def change_column_names(dataset):
    for colname in dataset.columns:
        dataset.rename(columns={colname: colname.replace(" ", "_")}, inplace=True)

def handle_duplicate_countries(dataframe, countries_to_change, countries_to_replace_with):
    fields_to_change = ["Country", "ISO", "Continent"]
    #Create empty list and counters
    fields_to_replace_with = []
    counter1, counter2 = 0, 0

    #Find and extract the correct fields that the countries to change should have
    for row in range(len(dataframe)):
        if dataframe["ISO"][row] in countries_to_replace_with:
            for field in fields_to_change:
                fields_to_replace_with.append(dataframe[field][row])
    fields_to_replace_with = list(dict.fromkeys(fields_to_replace_with))

    #Find and change the fields with the wrong values:
    for row in range(len(dataframe)):
        if dataframe["ISO"][row] in countries_to_change:
            for field in fields_to_change:
                dataframe.loc[((dataframe["ISO"] == "YMN") | (dataframe["ISO"] == "YMD")), field] = \
                fields_to_replace_with[3 + counter1]
                counter1 += 1
                dataframe.loc[((dataframe["ISO"] == "DFR") | (dataframe["ISO"] == "DDR")), field] = \
                fields_to_replace_with[counter2]
                counter2 += 1

def create_main_dataset(emdat):
    # Extract only the climate-driven, relevant disasters:
    relevant_disasters = ["Hydrological", "Meteorological", "Climatological"]
    temp_dataset = emdat[emdat.apply(lambda x: x["Disaster Subgroup"] in relevant_disasters, axis=1)]
    # Extract only the fatal events:
    main_dataset = temp_dataset[
        temp_dataset.apply(lambda x: not (pd.isnull(x["Entry Criteria"]) and pd.isnull(x["Total Deaths"])), axis=1)]
    return main_dataset.reset_index(drop=True)

def load_and_clean_EMDAT():
    #EMDAT global - EDIT ORIGINAL BASE DATASET for further usage
    emdat = pd.read_excel('emdat.xlsx', sheet_name='emdat data')
    print("Emdat length before filters: " + str(len(emdat)))
    #change_column_names(emdat)
    #Change Germany and Yemens old country names for the clustering
    countries_to_change = ["DFR", "DDR", "YMN", "YMD"]
    countries_to_replacewith = ["DEU", "DEU", "YEM", "YEM"]
    handle_duplicate_countries(emdat, countries_to_change, countries_to_replacewith)
    #Apply filters and get the fatal, climate-driven events
    base_dataset = create_main_dataset(emdat)
    base_dataset.to_excel("/Users/jannahovstad/PycharmProjects/Master//BASE_DF.xlsx")
    return base_dataset, emdat

def create_dataset_for_clustering(analysis_dataset):
    main_1987 = analysis_dataset[analysis_dataset["Year"] >1987]
    #main_dataset = create_main_dataset(emdat_1987)
    print("Length after 1987-year-filter: " + str(len(main_1987)))
    #print("Length after climate/fatal-filter: " + str(len(main_dataset)))
    #print(main_dataset["Disaster_Subgroup"].value_counts())
    return main_1987

def set_region_to_int(country_main_dataset):
    region_integer_dict = {'Northern America':0, 'Central America':1, 'Caribbean':2, 'South America': 3, 'Southern Africa':4, 'Middle Africa':5, 'Western Africa':6, 'Eastern Africa':7, 'Northern Africa':8, 'Western Asia':9,'Southern Asia':10, 'South-Eastern Asia':18, 'Australia and New Zealand':19, 'Polynesia':21, 'Melanesia':20, 'Micronesia':22, 'Eastern Asia':17, 'Central Asia':11, 'Russian Federation':16, 'Eastern Europe':12, 'Northern Europe':15,'Western Europe':14,'Southern Europe':13}
    for i in range(len(country_main_dataset)):
        country_main_dataset.loc[i, "Region"] = region_integer_dict[country_main_dataset["Region"][i]]
    country_main_dataset["Region"] = pd.to_numeric(country_main_dataset.Region)
    return country_main_dataset

def get_country_information(main_dataset):
    country_main_dataset = main_dataset[["Country", "ISO", "Region"]].drop_duplicates().reset_index(drop=True)
    country_main_dataset = set_region_to_int(country_main_dataset)
    return country_main_dataset

#Aggregate a given factor in a given way
def create_df_aggregate_by_country(dataset, aggcolumn, aggmethod, years=None, groupby_column="ISO"):
    if years != None:
        dataset = dataset.loc[(dataset['Year']>(2019-years)) & (dataset['Year']<2020)]
    disasters_per_country = dataset.groupby([groupby_column]).agg(
        {aggcolumn: aggmethod}).reset_index()
    disastercount_df = disasters_per_country[[groupby_column, aggcolumn]]
    if aggcolumn == "Dis No" and aggmethod == "count":
         disastercount_df.rename(columns={"Dis No": "#Disasters"}, inplace=True)
    if aggcolumn == "Total Deaths" and years == 5:
         disastercount_df.rename(columns={"Total Deaths": "Fatality 5-Year-Mean"}, inplace=True)
    return disastercount_df

def get_missing_values_by_factor(dataset, groupby_column, column_in_focus):
    counted_missing_data_df = dataset.set_index(groupby_column).isna().sum(level=0).reset_index()
    # sorted_missing_data = naOverview.sort_values(by=[column_in_focus], ascending=False)[[groupby_column, column_in_focus]].reset_index()
    missing_data_by_iso_for_variable = counted_missing_data_df[[groupby_column, column_in_focus]].rename(
        columns={column_in_focus: ("NA in " + column_in_focus)})
    return missing_data_by_iso_for_variable

def create_relative_NA_variable(dataset, groupby_column, column_in_focus):
    missing_data_by_iso_for_variable = get_missing_values_by_factor(dataset, groupby_column, column_in_focus)
    disaster_count_by_country = create_df_aggregate_by_country(dataset, aggcolumn="Dis No", aggmethod='count')
    disaster_count_and_missing_values_by_iso = missing_data_by_iso_for_variable.merge(disaster_count_by_country, on='ISO')
    disaster_count_and_missing_values_by_iso['Missing Data'] = disaster_count_and_missing_values_by_iso.iloc[:, 1] / disaster_count_and_missing_values_by_iso.iloc[:, 2]
    return disaster_count_and_missing_values_by_iso.drop(columns={"NA in Total Deaths"}).reset_index(drop=True)

def create_df_disastercount_per_disastertype(main_dataset):
    # Make dataframe with empty counts for each type of disaster within a country
    disastertypes = main_dataset["Disaster Subgroup"].unique()
    count_by_disastertype_df = pd.DataFrame(main_dataset.ISO.unique(), columns=["ISO"])
    for type in disastertypes:
        count_by_disastertype_df[type] = 0
    # Count the observations of disastertypes and add for the country in question
    for iso in count_by_disastertype_df.ISO:
        country_observations = main_dataset[main_dataset["ISO"]==iso]
        type_counts = country_observations["Disaster Subgroup"].value_counts(normalize=True)
        for key, value in type_counts.items():
            for type in disastertypes:
                if key == type:
                    count_by_disastertype_df.loc[(count_by_disastertype_df['ISO'] == iso),key] = value
    return count_by_disastertype_df[np.append(disastertypes, "ISO")]

def get_population_density_data():
    pd_rawdata = pd.read_csv('External_Data/LatLong_Intervals.txt', sep=";", skiprows=28, skipinitialspace=True)
    pd_df = pd_rawdata[["ISO3166A3", "population", "land_total"]].rename(columns={"ISO3166A3":"ISO", "population":"Population", "land_total":"Land Total"}).drop_duplicates()
    return pd_df

def get_gdp_data():
    gdp_rawdata = pd.read_excel('External_Data/GDP_PPP_DATA.xls').rename(columns={'Country Code': 'ISO'})
    gdp_DF = gdp_rawdata[['ISO','2015', '2016','2017', '2018','2019']].set_index("ISO")
    return gdp_DF

def handle_GDP_data(main_dataset):
    years = ["2015", "2016", "2017", "2018", "2019"]
    gdp_dataset = main_dataset[["2015", "2016", "2017", "2018", "2019"]]
    missing_ISO_by_year = {}
    for year in years:
        missing, missing_ISO, missingCountries, uniqueISO = [], [], [], []
        for row in range(len(main_dataset)):
            if np.isnan(np.sum(main_dataset[year][row])):
                missing.append(main_dataset["ISO"][row])
                if main_dataset.index[row] not in uniqueISO:
                    uniqueISO.append(main_dataset["ISO"][row])
        if year not in missing_ISO_by_year.keys():
            missing_ISO_by_year[str(year)] = missing
        # Check that ISO is in final DF and get country names
        for i in range(len(main_dataset)):
            if main_dataset["ISO"][i] in missing_ISO_by_year[year]:
                missing_ISO.append(main_dataset["ISO"][i])
                missingCountries.append(main_dataset["Country"][i])
        missing_ISO_by_year[year] = [missing_ISO, missingCountries]
    # Printing a list with the ISO values that have missing GDP values + a count of how many years between 2015-2019 that are missing
    overview_dataframe = pd.DataFrame()
    for ISO in uniqueISO:
        counter_ISO = 0
        for dict_list in list(missing_ISO_by_year.values()):
            isolist = dict_list[0]
            if ISO in isolist:
                counter_ISO += 1
        #Print an overview over the number of years that are missing a gdp value
        new_row = {'ISO': ISO, 'Missing Years': counter_ISO}
        overview_dataframe = overview_dataframe.append(new_row, ignore_index=True)
    return overview_dataframe

def remove_flawed_irrelevant_countries(overview, dataset, missingYears_threshold, max_dis_Count):
    dataset_overview = pd.merge(dataset, overview, on="ISO", how="left")
    count_missingdata, count_few_disasters = 0, 0
    dataset_after_removal = dataset_overview.copy()
    print("Length of dataset before removing the countries missing GDP values:" + " " + str(len(dataset)))
    print("Number of countries with missing GDP values needing to be handled before this method: " + str(
        dataset_overview["Missing Years"].notnull().sum()))
    for i in range(len(dataset_overview["ISO"])):
        if dataset_overview["Missing Years"][i] >= missingYears_threshold:
            count_missingdata += 1
            if dataset_overview["#Disasters"][i] <= max_dis_Count:
                # print("Removed due to disaster count " + str(max_dis_Count) + ": ")
                # print(dataset_overview.loc[i])
                dataset_after_removal.drop([i], axis=0, inplace=True)
    partition_missing_gda_data = count_missingdata / len(dataset_overview)
    partition_joint_criteria = count_few_disasters / len(dataset_overview)
    partition_few_disasters = count_few_disasters / count_missingdata
    print("New length of dataset after removing the countries missing GDP values:" + " " + str(
        len(dataset_after_removal)))
    print("Number of countries removed:" + str(len(dataset_overview) - len(dataset_after_removal)))
    print("Number of countries with one or more missing GDP values :" + str(
        dataset_after_removal["Missing Years"].notnull().sum()))
    print("Missing data: " + str(partition_missing_gda_data) + "\nMissing data and >15 disasters: " + str(
        partition_joint_criteria))
    return dataset_after_removal.reset_index()

def replace_missing_gdp_values(gdp_DF, main_dataset):
    years = ["2015", "2016", "2017", "2018", "2019"]
    selected_gdp_colums = gdp_DF[["2015", "2016", "2017", "2018", "2019"]]
    low_income_years = selected_gdp_colums.loc[selected_gdp_colums.index == "LIC"]
    upper_middle_income_years = selected_gdp_colums.loc[selected_gdp_colums.index == "UMC"]
    heavily_indebted_poor_country = selected_gdp_colums.loc[selected_gdp_colums.index == "HPC"]
    high_income_country = selected_gdp_colums.loc[selected_gdp_colums.index == "HIC"]
    least_developed_countries = selected_gdp_colums.loc[selected_gdp_colums.index == "LDC"]
    latin_america_and_caribbean = selected_gdp_colums.loc[selected_gdp_colums.index == "LAC"]
    main_dataset = main_dataset.drop(columns="index")
    for year in years:
        for row in range(len(main_dataset)):
            if main_dataset["ISO"][row] == "CUB":
                main_dataset.loc[row, year] = upper_middle_income_years[year][0]
            if main_dataset["ISO"][row] == "SOM":
                main_dataset.loc[row, year] = low_income_years[year][0]
                #main_dataset.loc[row, year] = heavily_indebted_poor_country[year][0]
            if main_dataset["ISO"][row] == "TWN":
                main_dataset.loc[row, year] = high_income_country[year][0]
            if main_dataset["ISO"][row] == "YEM":
                main_dataset.loc[row, year] = least_developed_countries[year][0]
            if main_dataset["ISO"][row] == "SSD":
                main_dataset.loc[row, year] = least_developed_countries[year][0]
            if main_dataset["ISO"][row] == "VEN":
                main_dataset.loc[row, year] = latin_america_and_caribbean[year][0]

    # removing North Korea
    main_dataset.drop(main_dataset[main_dataset["ISO"] == "PRK"].index, inplace=True)
    main_dataset.reset_index()
    return main_dataset

def create_GDP_mean_and_final_df(filtered_cluster_df):
    gdp_years = filtered_cluster_df[['2015', '2016', '2017', '2018', '2019']]
    filtered_cluster_df = filtered_cluster_df.drop(['2015', '2016', '2017', '2018', '2019'], axis=1).reset_index()
    # Finner mean GDP og kaller kolonnen GDP
    mean_gdp_DF = gdp_years.mean(axis=1, skipna=True).reset_index().rename(columns={0: "GDP"})
    cleansed_df = filtered_cluster_df.merge(mean_gdp_DF).drop(columns="index")
    print("Number of countries which need to be handeled :" +" "+ str(len(cleansed_df[cleansed_df[["GDP", "Land Total"]].isna().any(axis=1)])))
    print("Countries with missing values which needs to be fixed: ")
    print(cleansed_df[cleansed_df[["GDP", "Land Total"]].isna().any(axis=1)])
    return cleansed_df

def handle_pd_data(latlong_df):#Handle the countries missing land total value
    missing_land_area_list={"NER": 1266700, "TCD": 1259200, "ATG": 404, "MAC": 30.4, "SSD": 619745 }
    for row in range(len(latlong_df)):
        if latlong_df["ISO"][row] in missing_land_area_list.keys():
            latlong_df.loc[row, "Land Total"]=missing_land_area_list[latlong_df["ISO"][row]]
    print("Number of countries needing to be handled: " + str(len(latlong_df[latlong_df[["Land Total"]].isna().any(axis=1)])))
    return latlong_df


#*******************************PCA CLUSTER **************************************#

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
    return ev

def plot_correlations(df):
    plt.figure(figsize=(10, 8))
    #df.rename(columns={"DisType PC1":"DisType1","DisType PC2":"DisType2" })
    sns.heatmap(df.corr(), annot=False, cmap='Blues')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()

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
    plt.xlabel('Principal Component', fontsize=14)
    plt.ylabel('Proportion of Variance Explained (Eigenvalue)', fontsize=14)
    plt.grid(False)
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

def cumulative_variance_plot(df_std):
    pca = PCA().fit(df_std)
    PC_values = np.arange(pca.n_components_)+1
    plt.figure(figsize=(10,8))
    plt.plot(PC_values, np.cumsum(pca.explained_variance_ratio_), 'bo--', linewidth=2)
    #plt.title("Cumulative Explained Variance by Components for "+name)
    #plt.xlim(0.5,len(df_std[0]))
    plt.xlabel("Number of Components", fontsize=14)
    plt.ylabel("Cumulative Explained Variance", fontsize=14)
    plt.axhline(y=0.85, linewidth=1, color='r', alpha=0.5)
    plt.axhline(y=0.99, linewidth=1, color='r', alpha=0.5)
    plt.grid(False)
    plt.show()

def loadings_plot(pca, loadings, columns, name):
    plt.figure(figsize=(8, 8))
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
        #add_string = f" ({round(xi, 2)} {round(yi, 2)})"
        plt.text(loadings[i,0]*1.2,
                 loadings[i,1]*1.15, fontsize=10,
                 s=columns[i])# + add_string,

    plt.xlabel(f"Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)", fontsize=14)
    plt.ylabel(f"Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)", fontsize=14)
    plt.title('Variable factor map (PCA)'+name)
    plt.show()

def score_plot(x_pca, df, name):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x_pca[:, 0], x_pca[:, 1],
                    palette="Set1", legend='full', s=100).legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Principal Component 1 ', fontsize=14)
    plt.ylabel('Principal Component 2 ', fontsize=14)
    plt.axvline(0, ls='--')
    plt.axhline(0, ls='--')
    plt.title("Score plot-"+name)
    for i in range(x_pca.shape[0]):
        if x_pca[i, 0]>4 or x_pca[i, 0]<-3.5:
            plt.text(x=x_pca[i, 0], y=x_pca[i, 1], s=df.iloc[i, 0], fontsize=10)
    plt.grid(color='lightgray', linestyle='--')
    plt.show()

def biplot(score,coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    plt.figure(figsize=(10,8))
    plt.scatter(xs*scalex,ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color='#306754', fontweight='semibold',ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(0), fontsize=14)
    plt.ylabel("PC{}".format(1), fontsize=14)
    #plt.title("Biplot - "+name)
    plt.grid()
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
    plot_correlations(df)

def pca_analysis(df, name, names=None):
    df_std = StandardScaler().fit_transform(df)
    pca = PCA().fit(df_std)
    pca_scores = pca.transform(df_std)
    loadings = pca.components_
    print('\nPCA Eigenvalues: \n%s' % pca.explained_variance_)
    #print('Eigenvectors \n%s' % pca.components_)
    scree_plot(pca, name)
    create_explained_variance_plot(df_std, name)
    cumulative_variance_plot(df_std)
    loadings_plot(pca, loadings, df.columns, name)
    score_plot(pca_scores, names, name)
    biplot(pca_scores, loadings, labels=df.columns.values)

def investigate_nr_of_components(df, isos):
    #pca_analysis(df, "PCA for clustering", df)
    pca_analysis(df, "PCA for clustering", isos)
    factor_analysis(df,"PCA for clustering" )

def create_PCA_scores(df, nr_pca_components):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)
    pca = PCA(n_components=nr_pca_components, random_state=43)
    pca = pca.fit(df_std)
    pca_scores = pca.transform(df_std)
    return pca_scores, pca

def investigate_cluster_nr(min, max, pca_scores):
    rows,cols = 6, 6 #REMEMBER TO ADJUST
    fig, ax = plt.subplots(rows, cols, figsize=(15, 8))
    row, col, first = 0, 0, 0
    for i in range(min, max): #was 5, 30
        if first != 0:
            if col<cols-1:
                col+=1
            elif col==cols-1:
                col = 0
                row +=1
        first=1
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_scores)
        visualizer = SilhouetteVisualizer(kmeans_pca, colors='yellowbrick', ax=ax[row][col])
        visualizer.fit(pca_scores)

def get_silhouette_scores_by_clusternumber(min_range, max_range, pca_scores):
    clustersums, sil_score_tuples, sil_scores, clusternr_list = [], [], [], []
    for i in range(min_range, max_range):
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_scores)
        score = silhouette_score(pca_scores, kmeans_pca.labels_,metric='euclidean')
        sil_score_tuples.append([i, score])
    sil_score_tuples.sort(key=lambda x: x[0])  # print(sil_scores) to investigate nr of clusters through Silhouette scores.
    for tuple in sil_score_tuples:
        sil_scores.append(tuple[1])
        clusternr_list.append(tuple[0])
    return sil_scores, clusternr_list

def silhouette_plot_clusternumber(sil_scores, nrcluster_list):
    plt.figure(figsize=(10,8))
    plt.plot(nrcluster_list, sil_scores, marker='o', linestyle='--')
    plt.title("Silhouette analysis of number of clusters")
    plt.xlabel("Nr of clusters", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.show()

def inertia_plot_clusternumber(min, max):
    clustersums = []
    for i in range(min, max): #was 5, 30
        kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans_pca.fit(pca_scores)
        clustersums.append(kmeans_pca.inertia_)
    plt.figure(figsize=(10,8))
    plt.plot(range(min, max), clustersums, marker='o', linestyle='--')
    plt.xlabel("Number of clusters", fontsize=14)
    plt.ylabel("Inertia score", fontsize=14)
    plt.show()

def plot_selected_clusternumber_silhouette_scores(data, min, max, ):
    print("Tuning Silhouette: ")
    # candidate values for our number of cluster
    parameters = [18, 19, 20,23, 25, 28, 19, 30, 31, 32, 33, 34, 35, 40,41, 45, 46]
    parameters = range(min, max)
    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})
    best_score = -1
    kmeans_model = KMeans(random_state=42)  # instantiating KMeans model
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(data)  # fit model on wine dataset, this will find clusters based on parameter p
        ss = silhouette_score(data, kmeans_model.labels_)  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores
        #print('Parameter:', p, 'Score', ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    # plotting silhouette score
    plt.figure(figsize=(10,8))
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.4)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters', fontsize=14)
    plt.show()

def investigate_number_of_clusters(pca_scores, minrange, maxrange):
    #investigate_cluster_nr(minrange, maxrange, pca_scores) #Plott alle størrelsene sine silhouette verdier
    sil_scores, clusternr_list = get_silhouette_scores_by_clusternumber(minrange, maxrange, pca_scores)
    silhouette_plot_clusternumber(sil_scores, clusternr_list)
    inertia_plot_clusternumber(minrange, maxrange)
    #plot_selected_clusternumber_silhouette_scores(pca_scores, minrange, maxrange)

def add_PCA_and_segment_column(nr_pca_components, df, pca_scores, kmeans_pca):
    df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(pca_scores)], axis=1)
    componentnames = []
    for i in range(nr_pca_components):
        componentnames.append("Component" + str(i))
    df_segm_pca_kmeans.columns.values[-nr_pca_components:] = componentnames
    df_segm_pca_kmeans.loc[:,'Cluster Number'] = np.array(list(map(lambda x: x+1, kmeans_pca.labels_)))
    df_segm_pca_kmeans.loc[:,'Cluster Name'] = df_segm_pca_kmeans.loc[:,'Cluster Number'].apply(lambda x: 'Cluster ' + str(x))
    return df_segm_pca_kmeans, componentnames

def rand_col(nr_of_clusters, min_v=0.2, max_v=1.0):
    colorlist = []
    for i in range(1, nr_of_clusters+1):
        hex = "#FFFFFF"
        while hex not in colorlist:
            hsv = np.concatenate([np.random.rand(2), np.random.uniform(min_v, max_v, size=1)])
            hex = to_hex(hsv_to_rgb(hsv))
            colorlist.append(hex)
    return colorlist

def create_3D_plot(df):
    sns.set_style("white")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    cluster_list = [8,12,15,9,29,21,20,38]
    df2 = df[df.loc[:,'Cluster Number'].isin(cluster_list)]
    #df2.loc[:,'Cluster Name'] = pd.Categorical(df2.loc[:,'Cluster Number'])
    colors = {8: 'orange', 12: 'olive', 15: 'blue', 9: 'gold', 20: 'deepskyblue', 21:'red', 29:'turquoise', 38:'hotpink'}

    # Create scatter
    sc =ax.scatter(df2['Component0'], df2['Component1'], df2['Component2'], c= df2['Cluster Number'].map(colors), s=55, alpha=1)

    #Make simple, bare axis lines through space:
    xAxisLine = ((min(df2['Component0']), max(df2['Component0'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(df2['Component0']), max(df2['Component0'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(df2['Component1']), max(df2['Component1'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # labeling the axes
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.set_zlabel("PC3", fontsize=14)
    plt.legend(*sc.legend_elements(),bbox_to_anchor=(1.05, 1), loc=2, title='Cluster Number:')
    plt.show()

def plot_cluster_silhouette_scores(kmeans_pca, pca_scores):
    print("Running analyze Silhuette Scores")
    cluster_labels = kmeans_pca.labels_
    silhuette_avg = silhouette_score(pca_scores, cluster_labels)
    print("The average silhuette score is :", silhuette_avg)
    silhouette_values = silhouette_samples(pca_scores, cluster_labels)

    y_lower = y_upper = 0
    fig, ax = plt.subplots(figsize=(8, 9))
    for i, cluster in enumerate(np.unique(cluster_labels)):
        cluster_silhouette_vals = silhouette_values[cluster_labels == cluster]
        cluster_silhouette_vals.sort()
        if len(cluster_silhouette_vals)>1:
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1);
            ax.text(-0.03, (y_lower + y_upper) / 2, str(i+1), fontsize=10)
            y_lower += len(cluster_silhouette_vals)
            # Get the average silhouette score
            avg_score = np.mean(silhouette_values)
            ax.axvline(avg_score, linestyle='--', linewidth=2, color='green')
            ax.set_yticks([])
            ax.set_xlim([-0.1, 1])
            ax.set_xlabel('Silhouette coefficient values', fontsize=14)
            ax.set_ylabel('Cluster labels', fontsize=14)
            plt.tight_layout()

def plot_cluster_results(cluster_results, kmeans_pca, pca_scores):
    create_3D_plot(cluster_results)
    plot_cluster_silhouette_scores(kmeans_pca, pca_scores)

def analyze_cluster_results(cluster_results, nr_clusters, pca_scores, print_results=True):
    averages, clusternumbers, silhouettes = [], [], []
    cluster_labels = np.array(cluster_results['Cluster Number'])
    silhouette_values = silhouette_samples(pca_scores, cluster_labels) #Returns silhouette valkue for each sample

    for i in range(1,nr_clusters+1):
        current_cluster = cluster_results.loc[cluster_results['Cluster Number'] ==i]
        ith_cluster_silhouette_values =silhouette_values[cluster_labels == i]
        average = sum(ith_cluster_silhouette_values) /len(ith_cluster_silhouette_values)
        if print_results==True:
            print("Cluster " + str(i) + " contains " + str(len(current_cluster)) + " countries.")
            print(list(current_cluster["Country"]))
            print(list(current_cluster["Total Deaths"]))
            print("Total death MEAN: " + str((current_cluster["Total Deaths"].mean())))
            print("Total death MIN-MAX: " + str((current_cluster["Total Deaths"].min())) + " - " + str(
                (current_cluster["Total Deaths"].max())))
            print("Silhouette values for samples within cluster " + str(i) + " is: ")
            print(ith_cluster_silhouette_values)
            print("Average: " + str(average))
            print("\n")

        iso_and_silhouette = list(zip(list(current_cluster["ISO"]),ith_cluster_silhouette_values))
        silhouettes = silhouettes + iso_and_silhouette
        averages.append(round(average, 3))
        clusternumbers.append(i)

    iso_and_silhouette_df = pd.DataFrame(silhouettes, columns=['ISO', 'Silhouette Value'])
    Data = {'Cluster Number': clusternumbers,
                'Silhouette Average': averages}
    average_df = pd.DataFrame(Data)

    return average_df, iso_and_silhouette_df

def addCorrectCountryNames(dataset):
    new_countrynames = pd.read_excel('External_Data/countrynames.xls')[["Country PBI","ISO"]]
    new_countrynames["ISO"] = new_countrynames["ISO"].str.strip()
    dataset_merged = dataset.merge(new_countrynames, on="ISO", how="left")
    return dataset_merged

def create_country_general(country_dataset, main_dataset):
    country_df_w_names = addCorrectCountryNames(country_dataset)
    relevant = country_df_w_names[['ISO','Region', 'Missing Data', 'Land Total', 'Country PBI',
                           'Cluster Number', 'Cluster Name', 'Silhouette Average', 'Silhouette Value']]
    relevant.merge(main_dataset[["ISO", "Continent"]], on="ISO", how="left")
    return relevant

#MAIN:
if __name__ == "__main__":

    #********************************READ IN DATASETS******************************

    #Get and create dataframes from emdat
    BASE_DF, emdat = load_and_clean_EMDAT()

    #Modify and aggregate base dataset for clustering
    main_dataset = create_dataset_for_clustering(BASE_DF)
    country_info_df = get_country_information(main_dataset)
    relative_missing_values_df = create_relative_NA_variable(main_dataset, 'ISO', 'Total Deaths')
    fatality_five_year_mean = create_df_aggregate_by_country(main_dataset, aggcolumn='Total Deaths', aggmethod='sum', years=5)
    total_deaths_per_country_df = create_df_aggregate_by_country(main_dataset, aggcolumn='Total Deaths', aggmethod='sum')
    cpi_mean_per_country_df = create_df_aggregate_by_country(main_dataset, aggcolumn='CPI', aggmethod='mean')
    dis_count_by_distype_df = create_df_disastercount_per_disastertype(main_dataset[["ISO", "Disaster Subgroup"]])
    aggregated_emdat_data = [relative_missing_values_df, total_deaths_per_country_df, cpi_mean_per_country_df, dis_count_by_distype_df, fatality_five_year_mean]

    #Get and create external datasets
    pd_df = get_population_density_data()
    gdp_df = get_gdp_data()
    external_cluster_data =[pd_df, gdp_df]

    #Merge all databases:
    emdat_cluster_data = country_info_df
    for dataset in aggregated_emdat_data:
        emdat_cluster_data = emdat_cluster_data.merge(dataset, on="ISO", how="left")
    final_df = emdat_cluster_data
    for dataset in external_cluster_data:
        final_df = final_df.merge(dataset, on="ISO", how="left")

    # ***************************HANDELING NA VALUES in final_df****************r***********
    missing_gdp_iso = handle_GDP_data(final_df)
    roughly_filtered_cluster_df = remove_flawed_irrelevant_countries(missing_gdp_iso, final_df, 5, 15)
    filtered_cluster_df = replace_missing_gdp_values(gdp_df, roughly_filtered_cluster_df)
    cleansed_df = create_GDP_mean_and_final_df(filtered_cluster_df)
    cleansed_df = handle_pd_data(cleansed_df)
    cleansed_df["PD"] = cleansed_df.loc[:, 'Population'] / cleansed_df.loc[:, 'Land Total']
    #investigate_nr_of_components(cleansed_df[["Hydrological", "Meteorological", "Climatological"]])
    #pca_analysis(cleansed_df[["Hydrological", "Meteorological", "Climatological"]], " Disaster Types")
    #factor_analysis(cleansed_df[["Hydrological", "Meteorological", "Climatological"]], " Disaster Types")

    cleansed_df[["DisType PC1", "DisType PC2"]], pca = create_PCA_scores(cleansed_df[["Hydrological", "Meteorological", "Climatological"]], nr_pca_components=2)
    CLUSTER_DF = cleansed_df.drop(['Missing Years'], axis=1)
    CLUSTER_DF.to_excel("/Users/jannahovstad/PycharmProjects/Master//CLUSTER_DF.xlsx")

    # ************************************CLUSTERING*****************************************
    print("\n"+"CLUSTERING")
    isos = cleansed_df[["ISO"]]
    df = cleansed_df.drop(['Country', 'ISO', 'Population', 'Land Total', 'Missing Years', "CPI", "Hydrological", "Meteorological", "Climatological", "Total Deaths", "Fatality 5-Year-Mean"], axis=1)

    print("Length of final cluster dataset: " + str(len(df)))
    print("DF for clustering overview NA: ")
    print(df.isna().sum())
    print("\n" + "Final columns in cluster dataset: ")
    print(df.columns)
    print("\n")

    #PCA:
    investigate_nr_of_components(df, isos)
    nr_pca_components= 6
    pca_scores, pca = create_PCA_scores(df, nr_pca_components)

    #Clustering:
    investigate_number_of_clusters(pca_scores, minrange=15, maxrange=51)
    nr_of_clusters = 40
    kmeans_pca = KMeans(n_clusters=nr_of_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(pca_scores)
    df_segm_pca_kmeans, component_names = add_PCA_and_segment_column(nr_pca_components, df, pca_scores, kmeans_pca)
    cluster_results = df_segm_pca_kmeans.join(cleansed_df[["Country", "ISO", "Total Deaths", "Land Total"]])

    # EVALUATION
    plot_cluster_results(cluster_results, kmeans_pca, pca_scores)
    average_df, silhouettes_by_iso = analyze_cluster_results(cluster_results, nr_of_clusters, pca_scores)
    cluster_results = cluster_results.merge(average_df, on='Cluster Number', how='left').merge(silhouettes_by_iso, on="ISO", how="left")
    cluster_results.to_excel("/Users/jannahovstad/PycharmProjects/Master//cluster_PBI.xlsx")

    #cluster_results = addCorrectCountryNames(cluster_results)
    COUNTRY_GENERAL = create_country_general(cluster_results, main_dataset)
    COUNTRY_GENERAL.to_excel("/Users/jannahovstad/PycharmProjects/Master//COUNTRY_GENERAL.xlsx")


    # SILHOUETTE:
    #analyzeSilhuetteScores(kmeans_pca, pca_scores)

    ####################################### POWER BI #############################################

    # create pandas dataframe for Power BI
    resultdata_powerBI = pd.DataFrame(data=df_segm_pca_kmeans, columns=df_segm_pca_kmeans.columns)
    resultdata_powerBI = resultdata_powerBI.iloc[:, 7:].astype('str')

    # create pandas dataframe of the pca data for Power BI
    columns = np.append(df.columns, ['VARRATIO'])
    data = np.concatenate((pca.components_, pca.explained_variance_ratio_.reshape(-1, 1)), axis=1)
    df_pca = pd.DataFrame(data=data, columns=columns, index=component_names)
    df_pca = df_pca.astype('str')
