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


def changeColumnNames(dataset):
    for colname in dataset.columns:
        dataset.rename(columns={colname: colname.replace(" ", "_")}, inplace=True)

def handleDuplicateCountries(dataframe, countries_to_change, countries_to_replace_with):
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

def createMainDataset(emdat):
    # Extract only the climate-driven, relevant disasters:
    relevant_disasters = ["Hydrological", "Meteorological", "Climatological"]
    temp_dataset = emdat[emdat.apply(lambda x: x.Disaster_Subgroup in relevant_disasters, axis=1)]
    # Extract only the fatal events:
    main_dataset = temp_dataset[
        temp_dataset.apply(lambda x: not (pd.isnull(x["Entry_Criteria"]) and pd.isnull(x["Total_Deaths"])), axis=1)]
    main_dataset = main_dataset.reset_index().drop(columns=["index"])
    return main_dataset

def load_and_clean_EMDAT():
    #EMDAT global - EDIT ORIGINAL BASE DATASET for further usage
    emdat = pd.read_excel('emdat.xlsx', sheet_name='emdat data')
    changeColumnNames(emdat)
    #Change Germany and Yemens old country names for the clustering
    countries_to_change = ["DFR", "DDR", "YMN", "YMD"]
    countries_to_replacewith = ["DEU", "DEU", "YEM", "YEM"]
    handleDuplicateCountries(emdat, countries_to_change, countries_to_replacewith)
    #Apply filters and get the fatal, climate-driven events
    analysis_dataset = createMainDataset(emdat)
    return analysis_dataset, emdat

# PLOT
def standardPlot(xaxis, yaxis, x_label, y_label, rotation=70, plot_title=" ", bottom=0.45, size=(10, 8)):
    plt.figure(figsize=size).subplots_adjust(bottom=bottom)
    plt.plot(xaxis, yaxis)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)
    plt.show()

# NA ANALYSIS
def hasNA(dataset):
    na_vaules, xaxis, complete_columns = [], [], []
    for column in dataset.columns:
        if dataset[column].isnull().values.any():
            na_vaules.append(dataset[column].isnull().sum())
            xaxis.append(column)
        else:
            complete_columns.append(column)
    na_vaules.sort()
    standardPlot(xaxis, na_vaules, x_label="Columns with NA values", y_label="Number of NA values")
    return zip(xaxis, na_vaules), complete_columns

def generalExplorationCluster(within_cluster):
    print(within_cluster.Country.value_counts())
    print(len(within_cluster))
    zipobj, complete_cols = hasNA(within_cluster)
    return complete_cols

def getRelevantCountries():
    iso_list = list(pd.read_excel("COUNTRY_GENERAL.xlsx").ISO)
    return iso_list

def filterDataset_relevantCountries(dataset):
    relevant_countries = getRelevantCountries()
    correct_countries_dataset = dataset[dataset.apply(lambda x: x.ISO in relevant_countries, axis=1)]
    analysis_dataset = correct_countries_dataset.reset_index().drop(columns=["index"])
    return analysis_dataset

def addOutlierDetails(dataset):
    outlier_df = pd.read_excel("outlier_df.xlsx")
    #Extract relevant variables from dataset
    outlier_df = outlier_df[["Dis_No", 'Latitude', 'Longitude', 'Max_Wind', 'Max_Surge','Max_Waterheight', 'Date', 'Location', 'Event_Name']]
    dataset = dataset.merge(outlier_df, on="Dis_No", how="left")
    #Replace empty fields with fields in outlier df if they exist.
    duplicate_columns = ['Event_Name','Location', 'Latitude','Longitude']
    for column in duplicate_columns:
        dataset[column] = dataset.set_index("Dis_No")[column + str('_x')].fillna(
        dataset.set_index("Dis_No")[column + str('_y')]).reset_index()[column + str('_x')]
        dataset = dataset.drop(columns=[column+str('_x'), column+str('_y')])
    dataset["Event_Name"] = dataset["Event_Name"].fillna(dataset["Dis_No"]).reset_index()["Event_Name"]
    return dataset

def convert_month(df):
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df["Month_Name"] = df["Start_Month"].replace(month_dict)
    return df

def create_Disaster_Dataset(analysis_dataset):
    #Retrieve the observations from the countries in the dataset after filtering
    correct_countries_dataset = filterDataset_relevantCountries(analysis_dataset)
    #Drop the country variable from EMDAT
    disaster_dataset = correct_countries_dataset.drop(columns=["Country", "Seq", "Region"])
    #Make the datatype of Year integer
    disaster_dataset['Year'] = disaster_dataset['Year'].astype('int')
    # Add observation with total deaths and number of disasters in the world (WRD)
    disaster_dataset = add_total_worldsum_row(disaster_dataset)
    # Turn month numbers into names:
    disaster_dataset = convert_month(disaster_dataset)
    #Add general country information for all countries
    disaster_dataset = add_Country_General_Data(disaster_dataset)
    disaster_dataset = addOutlierDetails(disaster_dataset)

    return disaster_dataset

def check_data_availability(datasets):
    clusterlist = ['BGD', 'DEU', 'FRA', 'JPN', 'KOR', 'VNM']
    for name, df in datasets.items():
        print("\n")
        print(name)
        countries_present =[]
        for iso in clusterlist:
            isolist = []
            for index in list(range(4,65)):
                relevant_row = df[df.ISO == iso]
                if (not np.isnan(relevant_row.iloc[0,index])):
                    isolist.append(relevant_row.columns[index])
            print(iso)
            print(isolist)
    print("Length of dataset for within cluster analysis: " + str(len(analysis_dataset)))

def extractRelevantCountriesFromDict(dict):
    #Extract the observations from the 189 countries
    for name, dataset in dict.items():
        relevant_dataset = filterDataset_relevantCountries(dataset)
        d1 = {name: relevant_dataset}
        dict.update(d1)
    return dict


def read_datasets():
    base_df = pd.read_excel("CLUSTER_DF.xlsx")

    #Geography and landscape
    #low_areas = pd.read_excel("elevation_below_5.xls")
    #low_areas_rural = pd.read_excel("elevation_rural.xls")
    #low_areas_urban = pd.read_excel("elevation_urban.xls")
    #agricultural_land = pd.read_excel("agricultural_land.xls")

    #Population:
    PD =  pd.read_excel('Population_Density.xls')
    rural_population = pd.read_excel('RuralPopulation_percentage.xls')
    urban_population = pd.read_excel('UrbanPopulation_percentage.xls')
    age_0_14 = pd.read_excel('Age_0_14.xls')
    age_15_64 = pd.read_excel('Age_15_64.xls')
    age_65plus = pd.read_excel('Age_65_plus.xls')
    gender_distribution = pd.read_excel('Gender_Distribution.xls')
    age_dependency_ratio= pd.read_excel('age_dependency_ratio.xls')
    population_growth = pd.read_excel('population_growth.xls')
    coastal_population = pd.read_excel('coastal_population.xls')

    #Economy
    gdp = pd.read_excel('GDP_PPP_DATA.xls').rename(columns={'Country Code': 'ISO'})
    gdp_per_cap = pd.read_excel('GDP_per_cap.xls')

    gini = pd.read_excel('GINI_index.xls')
    income = pd.read_excel('per_cap_income.xls')
    poverty_headcount = pd.read_excel('Poverty_headcount.xls')

    #Education
    mean_years_schooling = pd.read_csv('mean-years-of-schooling.csv').drop(columns={'Entity'})
    tertiary_education = pd.read_csv('tertiary_education_population.csv').drop(columns={'Entity'})

    #EMployment
    unemployed = pd.read_excel('unemployed.xls')

    #Health
    healthcare_index = pd.read_csv('healthcare-access-and-quality-index.csv').rename(columns={'Code': 'ISO','HAQ Index (IHME (2017))':'HAQ_Index' }).drop(columns={'Entity'})
    health_expenditure = pd.read_excel('health_expenditure_gdp.xls')

    #Gather all relevant external datasets in a dictionary
    world_bank_datasets = {'Population Density': PD, 'Age Dependency': age_dependency_ratio, 'Population Growth':population_growth,
                "Rural Population": rural_population, "Urban Population":urban_population, 'Coastal Population': coastal_population,
                "Age Distribution 0-14":age_0_14, "Age Distribution 15-64":age_15_64, "Age Distribution 65+":age_65plus,
                #'Low Elevation Areas':low_areas, 'Low Elevation RURAL':low_areas_rural, 'Low Elevation URBAN':low_areas_urban,
                'GDP': gdp,'GDP per capita':gdp_per_cap, 'GINI':gini, 'Per Capita Income': income, 'Poverty Headcount': poverty_headcount,
                'Unemployment Rate': unemployed, 'Health Expenditure': health_expenditure}
    #check_data_availability(world_bank_datasets)

    external_datasets = {'Health Index':healthcare_index, 'Mean Years of Schooling': mean_years_schooling, "Teritary Education": tertiary_education}

    #Extract the observations from the 189 countries
    world_bank_datasets = extractRelevantCountriesFromDict(world_bank_datasets)
    external_datasets = extractRelevantCountriesFromDict(external_datasets)

    return world_bank_datasets, external_datasets

def create_Country_Yearly(emdat_yearly, datasets, other_datasets):
    #Retrieving the ISO column, as well as all the columns representing 1960 to 2020.
    columns_world_bank_dataset = [1] + list(range(4,65))
    yearly_country_data = emdat_yearly
    for name, dataset in datasets.items():
        word_bank_data = dataset.iloc[:, columns_world_bank_dataset]
        exploded_world_bank = pd.melt(word_bank_data, id_vars=['ISO'], var_name="Year", value_name=name)
        exploded_world_bank['Year'] = exploded_world_bank['Year'].astype(int)
        yearly_country_data=yearly_country_data.merge(exploded_world_bank, on=["ISO", "Year"], how="outer")
    for name, dataset in other_datasets.items():
        yearly_country_data=yearly_country_data.merge(dataset, on=["ISO", "Year"], how="outer")

    #Add constant country information such as Name, Land area etc.
    yearly_country = add_Country_General_Data(yearly_country_data)
    return yearly_country

def add_Country_General_Data(dataset):
    COUNTRY_GENERAL = pd.read_excel('COUNTRY_GENERAL.xlsx')
    dataset = dataset.merge(COUNTRY_GENERAL, on="ISO", how="left")
    return dataset

def add_yearly_worldsum_row(dataset):
    sum_deaths = dataset.groupby(["Year"]).agg({"Yearly_Fatality_Sum": "sum", "Yearly_Disaster_Count":"count"}).reset_index()
    sum_deaths["ISO"] = "WRD"
    dataset = dataset.append(sum_deaths)
    return dataset

def add_total_worldsum_row(dataset):
    dataset["Disaster_Count"] = 1
    aggregated_disasters = dataset.agg({"Total_Deaths": "sum", "Disaster_Count":"count"})
    world_deaths, world_eventcount = aggregated_disasters[0], aggregated_disasters[1]
    dataset = dataset.append(pd.DataFrame([[world_deaths, world_eventcount, 'WRD']], columns=["Total_Deaths", "Disaster_Count", "ISO"]))
    return dataset

def create_Disasters_Yearly(DISASTER_DF):
    aggregated_yearly_df = DISASTER_DF.groupby(['ISO', 'Year']).agg({"Total_Deaths":'sum', "Dis_No":'count'}).reset_index()
    aggregated_yearly_df.rename(columns={'Total_Deaths': 'Yearly_Fatality_Sum','Dis_No':'Yearly_Disaster_Count'}, inplace=True)
    #Create aggregated row for the total in the world per year
    yearly_disasterdata = add_yearly_worldsum_row(aggregated_yearly_df)
    yearly_disasterdata['Year'] = yearly_disasterdata['Year'].astype('int')
    return yearly_disasterdata

#MAIN:
if __name__ == "__main__":
    #Load disaster observations from EM-DAT
    analysis_dataset, emdat = load_and_clean_EMDAT()

    #Create disaster event dataset
    DISASTER_DF = create_Disaster_Dataset(analysis_dataset)
    DISASTER_DF.to_excel("/Users/jannahovstad/Documents//DISASTER_DF.xlsx")


    #Create country and year dataset:
    emdat_yearly = create_Disasters_Yearly(DISASTER_DF)
    world_bank_datasets, other_datasets = read_datasets()
    COUNTRY_YEARLY = create_Country_Yearly(emdat_yearly, world_bank_datasets, other_datasets)
    COUNTRY_YEARLY.to_excel("/Users/jannahovstad/Documents//COUNTRY_YEARLY.xlsx")

    #complete_columns = generalExplorationCluster(within_cluster)
    #investigateNAbyCountry(within_cluster, complete_columns)








#DELETED FROM createVulnerabilityIndex:

#TESTING MULTIBLOCK PCA
# population_component = cluster_2010[['Population Density','Age Dependency', 'Population Growth', 'Rural Population', 'Urban Population','Coastal Population']]
# economy_component = cluster_2010[['GDP', 'GINI', 'Per Capita Income', 'Poverty Headcount']]
# disaster_component = cluster_2010[['Yearly_Fatality_Sum', 'Yearly_Disaster_Count']]
# education_component = cluster_2010[['Unemployment Rate', 'AVG Years of Schooling', 'Population with tertiary schooling']]
# health_component = cluster_2010[['Health Expenditure', 'HAQ_Index']]
# isoframe = cluster_2010["ISO"].reset_index(drop=True)
# components = [population_component, economy_component, disaster_component, education_component, health_component]
# i = 0
# for component in components:
#     name = "Vulnerability Index " + str(i)
#     isoframe[name], pca = createPCAScores(component, nr_pca_components=1)
#     i += 1
# print(isoframe)


