##################################################
# Airbnb Price Prediction Project
##################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

#############################################
# Business Problem
#############################################

# The main purpose of the collection of datasets is to predict accommodation prices, analyze and estimate listing prices based on region or other features
# Before prediction, we are going to make exploratory data analysis and understand the dataset by using some visualization tools
# After we've done that, we are going to create new features and show importance scores, edit features based on feature importance on the dataset
# ML algorithms will come after beyond that and predicted values calculation will be made.

##############################################
# Dataset
#############################################

# US dataset has been collected from multiple datasets found on the Airbnb website.
# Different cities which part of the USA are in the dataset and Airbnb accommodation adverts
# that were published have different types of information like room type, hostname, availability date, etc.
# The dataset contains 17 columns and approximately 226k rows.

# Variables:

# id: unique listing id
# name: name of the listing
# host_id: unique host Id
# host_name: name of the host
# neighbourhood_group: group in which the neighborhood lies
# neighborhood: name of the neighborhood
# latitude: latitude of listing
# longitude: longitude of listing
# room_type: room type
# price: price of listing per night
# minimum_nights: minimum no. of nights required to book.
# number_of_reviews: total number of reviews on listing
# last_review: date on which listing received its last review
# reviews_per_month: average reviews per month on listing
# calculated_host_listings_count: total number of listings by host
# availability_365: number of days in year the listing is available for rent
# city: region of the listing


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
import plotly.graph_objects as go
import plotly.offline as py
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from wordcloud import WordCloud, STOPWORDS
# linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# non-linear models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import KNNImputer

import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from yellowbrick.cluster.elbow import kelbow_visualizer

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime as dt
from helpers.data_prep import *
from helpers.eda import *
import pickle
import sys
import warnings
import re

if not sys.warnoptions:
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



############################################
# Let's get the dataset
############################################

def load_airbnb():
    data = pd.read_csv("datasets/AB_US_2020.csv")
    return data


df = load_airbnb()
df.head()



################################################
# 1. Exploratory Data Analysis
################################################

df.describe().T

check_df(df)

"""----------------------------------------"""

# Frequencies are visually
sns.countplot(x="price", data=df)
plt.show()

"""----------------------------------------"""

# Frequencies are visually
df["availability_365"].hist()

"""----------------------------------------"""

# Class ratios of the target variable:
100 * df["price"].value_counts() / len(df)


##########################
# Plots
##########################

st_count = df['neighbourhood_group'].value_counts()
sns.set(style="darkgrid")
sns.barplot(st_count.values, st_count.index, alpha=0.9)
plt.title('Frequency of States')
plt.ylabel('State', fontsize=10)
plt.xlabel('Occurrences', fontsize=12)
plt.show()

"""----------------------------------------"""

df.groupby('price').mean()

"""----------------------------------------"""

df['room_type'].value_counts().plot(kind='barh', figsize=(6, 4),
                                    edgecolor=(0, 0, 0), color='tan', title='Room Type')

"""----------------------------------------"""

df['neighbourhood_group'].value_counts().plot(kind='barh', figsize=(6, 6),
                                              edgecolor=(0, 0, 0), color='lightblue', title='State')

"""----------------------------------------"""

df.plot(x='price', y='availability_365', style='+', color='salmon')
plt.xlabel('SalePrice')
plt.ylabel('availability_365')
plt.show()

"""----------------------------------------"""

plot_ = sns.catplot(x="neighbourhood_group", y="price", hue="room_type", kind="swarm", data=df)
plot_


##########################
# Variable types indicating
##########################


df1 = df.copy()
df1.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df1)

for col in cat_cols:
    cat_summary(df1, col, plot=True)

numerical_cols = [col for col in num_cols if col not in ['id', 'host_id', 'latitude', 'longitude']]

for col in numerical_cols:
    num_summary(df1, col, plot=True)

for col in cat_cols:
    target_summary_with_cat(df1, "price", col)


##########################
# Outlier Analysis
##########################

boxplot = df1.boxplot(figsize=(8, 7), rot=45)

"""----------------------------------------"""

outlier_thresholds(df1, df1.columns)
print({"Cat_Cols:": len(cat_cols), "Num_cols: ": len(numerical_cols), "Cat_but_Car: ": len(cat_but_car)})

"""----------------------------------------"""

for col in numerical_cols:
    print(col, check_outlier(df1, col))

"""----------------------------------------"""

sns.boxplot(x=df1["price"])
plt.show()

"""----------------------------------------"""

sns.distplot(df1["price"], hist=True)
df1["price"] = np.log1p(df1["price"])

"""----------------------------------------"""

sns.distplot(df["minimum_nights"], hist=True)
df1["minimum_nights"] = np.log1p(df1["minimum_nights"])

"""----------------------------------------"""

sns.distplot(df1["calculated_host_listings_count"], hist=True)
df1["calculated_host_listings_count"] = np.log1p(df1["calculated_host_listings_count"])

"""----------------------------------------"""

df1.plot(kind='density', subplots=True, layout=(14, 1), sharex=False, figsize=(10, 10))
plt.show()



##########################
# Missing Value Analysis
##########################

df1.isnull().sum().sum()

msno.bar(df1)

NA_col = df1.isnull().sum()
# NA_col = NA_col[NA_col.values >(0.3*len(df))]
plt.figure(figsize=(20, 4))
NA_col.plot(kind='bar')
plt.title('List of Columns & NA counts')
plt.show()


def filling_with_missing(dataframe, col_name, date=False):
    if date:
        dataframe[col_name] = dataframe[col_name].fillna("01/01/1990")

    else:
        dataframe[col_name] = dataframe[col_name].fillna("missing")


missing_values_table(df1)
filling_name = ["name", "host_name"]
filling_with_missing(df1, filling_name)
filling_with_missing(df1, "last_review", date=True)


"""-------------------- MISSING VALUES --------------------"""

df1['pricelog_'] = np.log(df1['price'])
df1["minimum_nights"] = np.log(df1["minimum_nights"])
df1["number_of_reviews"] = np.log(df1["number_of_reviews"])
df1["calculated_host_listings_count"] = np.log(df1["calculated_host_listings_count"])

df1['pricelog_'].replace([np.inf, -np.inf], np.nan, inplace=True)
df1['pricelog_'].fillna(df1['pricelog_'].mean(), inplace=True)

df1['minimum_nights'].replace([np.inf, -np.inf], np.nan, inplace=True)
df1['minimum_nights'].fillna(df1['minimum_nights'].mean(), inplace=True)

df1['number_of_reviews'].replace([np.inf, -np.inf], np.nan, inplace=True)
df1['number_of_reviews'].fillna(df1['number_of_reviews'].mean(), inplace=True)

df1['calculated_host_listings_count'].replace([np.inf, -np.inf], np.nan, inplace=True)
df1['calculated_host_listings_count'].fillna(df1['calculated_host_listings_count'].mean(), inplace=True)

##########################
# 2. Data Preprocessing & Feature Engineering
##########################

def preprocess_name(rows):
    sentence = str(rows).lower()
    sentence = re.sub('apt', 'apartment', sentence)
    sentence = re.sub('w/', 'with', sentence)
    sentence = re.sub('br', 'bedroom', sentence)
    sentence = re.sub('bedrms', 'bedroom', sentence)
    sentence = re.sub('blck', 'block', sentence)
    sentence = re.sub('univs', 'university', sentence)
    sentence = re.sub('&', 'and', sentence)
    sentence = re.sub('[+-\/|]', ' ', sentence)
    #sentence = re.sub('\s+', ' ',sentence)
    sentence = re.sub('\'', '', sentence)
    sentence = re.sub('‚òö', '', sentence)
    sentence = re.sub('[!#\"~*)(,.:;?]', ' ', sentence)
    sentence = "".join(re.findall('[a-zA-Z0-9\s]', sentence))
    sentence = re.sub('\s+', ' ',sentence)
    return sentence

"""----------------------------------------"""

# time difference
today_date = dt.datetime(2020, 12, 11)
df1['last_review'] = pd.to_datetime(df1['last_review'])
df1['last_review'].max()
df1['daydiff'] = (today_date.year - df1['last_review'].dt.year) * 12 + today_date.month - df1['last_review'].dt.month
df1["daydiff"].value_counts()

"""----------------------------------------"""

# top 5 accommodation place according to review date
df1["daydiff"].sort_values(ascending=False).head()

"""----------------------------------------"""

# data on days of the week
df1['day_name'] = df1['last_review'].dt.day_name()
df1.groupby("day_name").agg({"price": "mean"})

"""----------------------------------------"""

# word counts in name variable

df1['name'] = df1['name'].apply(preprocess_name)
df1["count_name"] = df1["name"].apply(lambda x: len(x.strip().split(" ")))

df1.groupby("count_name").agg({"price": "mean"})

"""----------------------------------------"""

# most used word counts
df1["count_name"].sort_values(ascending=False).head(10)

"""----------------------------------------"""

# 10 most expensive accommodation places and the number of words used
df1.groupby("count_name").agg({"price": "mean"}).sort_values(by="price",ascending=False).head(10)

"""----------------------------------------"""

# counting availibilty as monthly
df1["avaibility_month"] = (df1['availability_365'] / 30).astype(int)
df1.groupby("avaibility_month").agg({"price": "mean"})


"""----------------------------------------"""

# state and region information

# creating state names variables based on cities
df1.loc[(df1['city'] == 'New York City') , 'state_postal_code'] = 'NY'
df1.loc[(df1['city'] == 'Los Angeles') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'Hawaii') , 'state_postal_code'] = 'HI'
df1.loc[(df1['city'] == 'San Diego') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'Broward County') , 'state_postal_code'] = 'FL'
df1.loc[(df1['city'] == 'Austin') , 'state_postal_code'] = 'TX'
df1.loc[(df1['city'] == 'Clark County') , 'state_postal_code'] = 'NV'
df1.loc[(df1['city'] == 'Washington D.C.') , 'state_postal_code'] = 'WA'
df1.loc[(df1['city'] == 'San Clara Country') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'San Francisco') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'Seattle') , 'state_postal_code'] = 'WA'
df1.loc[(df1['city'] == 'Twin Cities MSA') , 'state_postal_code'] = 'MN'
df1.loc[(df1['city'] == 'New Orleans') , 'state_postal_code'] = 'LA'
df1.loc[(df1['city'] == 'Chicago') , 'state_postal_code'] = 'IL'
df1.loc[(df1['city'] == 'Nashville') , 'state_postal_code'] = 'TN'
df1.loc[(df1['city'] == 'Portland') , 'state_postal_code'] = 'OR'
df1.loc[(df1['city'] == 'Denver') , 'state_postal_code'] = 'CO'
df1.loc[(df1['city'] == 'Rhode Island') , 'state_postal_code'] = 'RI'
df1.loc[(df1['city'] == 'Boston') , 'state_postal_code'] = 'MA'
df1.loc[(df1['city'] == 'Oakland') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'San Mateo County') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'Jersey City') , 'state_postal_code'] = 'NJ'
df1.loc[(df1['city'] == 'Asheville') , 'state_postal_code'] = 'NC'
df1.loc[(df1['city'] == 'Santa Cruz County') , 'state_postal_code'] = 'CA'
df1.loc[(df1['city'] == 'Columbus') , 'state_postal_code'] = 'OH'
df1.loc[(df1['city'] == 'Cambridge') , 'state_postal_code'] = 'OH'
df1.loc[(df1['city'] == 'Salem') , 'state_postal_code'] = 'MA'
df1.loc[(df1['city'] == 'Pacific Grove') , 'state_postal_code'] = 'CA'


list1 = df1["state_postal_code"].value_counts().index
list1e = list1.tolist()

list2 = ["California", "New York", "Hawaii", "Washington", "Florida", "Texas", "Nevada", "Minnesota", "Louisiana",
         "Illinois", "Tennessee", "Oregon", "Colorado", "Rhode Island", "Massachusetts",
         "New Jersey", "Ohio", "North Carolina"]

dicts = {k: v for k, v in zip(list1e, list2)}

for state, name in dicts.items():
        df1.loc[(df1['state_postal_code'] == state) , 'state_name'] = name

"""----------------------------------------"""

# precious sortings...
df1.groupby("state_name").agg({"price" : "mean"}).sort_values(by="price",ascending=False)
df1.groupby("city").agg({"price" : "mean"}).sort_values(by="price",ascending=False)
df1.groupby("neighbourhood_group").agg({"price" : "mean"}).sort_values(by="price",ascending=False)
df1.groupby("neighbourhood").agg({"price" : "mean"}).sort_values(by="price",ascending=False)

df1["state_postal_code"].value_counts()


##########################
# AB Testing
##########################

# We are gonna check effects of entire home and other options on prices

# """------------------Normalization Assumption----------------------"""

# "H0: Normal distribution
# "H1: Not normal distribution
# pvalue < 0.05 than H0 is rejected. Because of that we don't need to look variance homogenity.

# test_stat, pvalue = shapiro(df1.loc[df1["room_type"] == "Entire home/apt", "price"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# test_stat, pvalue = shapiro(df1.loc[df1["room_type"] != "Entire home/apt", "price"])
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# >>> Assumption are not related. So non-parametric test will be used.

# test_stat, pvalue = mannwhitneyu(df1.loc[df1["room_type"] == "Entire home/apt", "price"],
#                            df1.loc[df1["room_type"] != "Entire home/apt", "price"])

# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# pvalue < 0.05, by this condition, test result tells us to entire home prices and other option prices have no similar effects.
# HO thesis is rejected

# df.groupby("room_type").agg({"price":"mean"})

##########################
# Segmentation with K-means
##########################

col_name = ["calculated_host_listings_count"]
rfmkmeans = df1[col_name]

sc = MinMaxScaler((0, 1))
x_sc = sc.fit_transform(rfmkmeans)
rfm_sc = pd.DataFrame(x_sc)
rfm_sc.head()

# Determining optimum cluster number
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfm_sc)
elbow.show()

# results of how many cluster should be
elbow.elbow_value_


# Creating final clusters
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm_sc)
kumeler = kmeans.labels_


pd.DataFrame({"id": rfmkmeans.index, "Segments": kumeler})
df1["cluster_no"] = kumeler


df1.groupby("cluster_no")["state_name"].value_counts()

##########################
# World Cloud and New Features
##########################

nltk.download('wordnet')

df1["name"] = df1["name"].str.replace('[^\w\s]', '')
df1["name"] = df1["name"].str.replace('\d', '')
df1["name"] = df1["name"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


comment_words = ''
stopwords = set(STOPWORDS)
for val in df1.name:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

plt.figure(figsize=(8, 12), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')

df1["unigrams"] = df1["name"].apply(nltk.word_tokenize)
df1["unigrams"] = df1["unigrams"].apply(lambda x: [item for item in x if item not in stopwords.words('english')])

result = pd.Series(np.concatenate([x for x in df1.unigrams])).value_counts()
result = pd.DataFrame({'ngrams': list(result.keys()),'count': list(result[:])})

result = result[result.ngrams != '.']
result = result.head(100)
result1 = result['ngrams'][1:100]
result1


def inner(text):
    for word in text.split(' '):
        for c in result1:
            if word == c:
                return c
    return 'Null'


c1 = []
for text in df1['name']:
    c1.append(inner(text))
df1['Name1'] = c1
df1.head()


plt.figure(figsize=(20, 4))
df1_group[1:20].plot(kind='bar')
plt.title('List of Columns Word counts')
plt.show()


names = ['Name1', 'price']

for column_name in names[:-1]:
    cd = ['ocean', 'beach', 'Null', 'cozy', 'room']
    cd1 = pd.DataFrame(cd, columns=['label'])
    s = cd1['label']

    for s1 in s:
        # Subset to the airline
        subset = df1[df1[column_name] == s1]

        # Draw the density plot
        ax = sns.distplot(subset['price'], hist=False, kde=True,
                          kde_kws={'shade': True, 'linewidth': 2},
                          label=s1).set(xlim=(0))


# Drop columns have no meaning now after feature engineering, because new features also carrying informations
drop_list = ["name","host_name","id","host_id","last_review","neighbourhood_group","neighbourhood","latitude","longitude","last_review",
             "city","state_name","reviews_per_month","unigrams"]


df1.drop(drop_list,inplace=True,axis=1)


##########################
# Encoding and Scaling
##########################

# Label Encoder
cat_cols, num_cols, cat_but_car = grab_col_names(df1)

labeller = ["room_type","day_name"]

for col in labeller:
    df1 = label_encoder(df1, col)

# One-Hot Encoding

df1.head()
ohe_cols = ["state_postal_code","Name1","cluster_no"]

df1 = one_hot_encoder(df1, ohe_cols)

df1.columns
df1.head()

# Min Max Scaler
cat_cols, num_cols, cat_but_car = grab_col_names(df1)
numerical_cols = [col for col in num_cols if "price" not in col]

ms = MinMaxScaler()
df1[numerical_cols] = ms.fit_transform(df1[numerical_cols])


cat_cols, num_cols, cat_but_car = grab_col_names(df1)

# Target and independent variables creating
y = df1["pricelog_"]
X = df1.drop(["price", "pricelog_"], axis=1)



# list feature importance for a regressor model like LGBM
def plot_importance(model, features, num=50, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Feature", y="Value", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.xticks(rotation=90)
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

pre_model = LGBMRegressor().fit(X, y)
feature_imp = pd.DataFrame({'Feature': X.columns, 'Value': pre_model.feature_importances_})
feature_imp.sort_values("Value", ascending=False)

plot_importance(pre_model,X)

df1.head()

# after than, we turns to function all we did above.

def airbnb_data_prep(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    numerical_cols = [col for col in num_cols if col not in ['id', 'host_id', 'latitude', 'longitude']]

    """-------------------- OUTLIER --------------------"""
    dataframe['pricelog_'] = np.log(dataframe['price'])
    dataframe["minimum_nights"] = np.log(dataframe["minimum_nights"])
    dataframe["number_of_reviews"] = np.log(dataframe["number_of_reviews"])
    dataframe["calculated_host_listings_count"] = np.log(dataframe["calculated_host_listings_count"])

    dataframe['pricelog_'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe['pricelog_'].fillna(dataframe['pricelog_'].mean(), inplace=True)

    dataframe['minimum_nights'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe['minimum_nights'].fillna(dataframe['minimum_nights'].mean(), inplace=True)

    dataframe['number_of_reviews'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe['number_of_reviews'].fillna(dataframe['number_of_reviews'].mean(), inplace=True)

    dataframe['calculated_host_listings_count'].replace([np.inf, -np.inf], np.nan, inplace=True)
    dataframe['calculated_host_listings_count'].fillna(dataframe['calculated_host_listings_count'].mean(), inplace=True)


    """ --------------------MISSING-------------------- """
    filling_name = ["name", "host_name"]

    filling_with_missing(dataframe, filling_name)
    filling_with_missing(dataframe, "last_review", date=True)

    """-------------------- FEATURE ENGINEERING --------------------"""
    today_date = dt.datetime(2020, 12, 11)
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    dataframe['last_review'].max()
    dataframe['daydiff'] = (today_date.year - dataframe['last_review'].dt.year) * 12 + today_date.month - dataframe[
        'last_review'].dt.month

    # haftanın günlerine ilişkin veriler
    dataframe['day_name'] = dataframe['last_review'].dt.day_name()

    # name değişkenindeki kelime sayısı
    dataframe['name'] = dataframe['name'].apply(preprocess_name)
    dataframe["count_name"] = dataframe["name"].apply(lambda x: len(x.strip().split(" ")))


    # ay sayılarının çıkarılması
    dataframe["availability_month"] = (dataframe['availability_365'] / 30).astype(int)

    # city bilgisi ve bölge bilgisi
    dataframe.loc[(dataframe['city'] == 'New York City'), 'state_postal_code'] = 'NY'
    dataframe.loc[(dataframe['city'] == 'Los Angeles'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'Hawaii'), 'state_postal_code'] = 'HI'
    dataframe.loc[(dataframe['city'] == 'San Diego'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'Broward County'), 'state_postal_code'] = 'FL'
    dataframe.loc[(dataframe['city'] == 'Austin'), 'state_postal_code'] = 'TX'
    dataframe.loc[(dataframe['city'] == 'Clark County'), 'state_postal_code'] = 'NV'
    dataframe.loc[(dataframe['city'] == 'Washington D.C.'), 'state_postal_code'] = 'WA'
    dataframe.loc[(dataframe['city'] == 'San Clara Country'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'San Francisco'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'Seattle'), 'state_postal_code'] = 'WA'
    dataframe.loc[(dataframe['city'] == 'Twin Cities MSA'), 'state_postal_code'] = 'MN'
    dataframe.loc[(dataframe['city'] == 'New Orleans'), 'state_postal_code'] = 'LA'
    dataframe.loc[(dataframe['city'] == 'Chicago'), 'state_postal_code'] = 'IL'
    dataframe.loc[(dataframe['city'] == 'Nashville'), 'state_postal_code'] = 'TN'
    dataframe.loc[(dataframe['city'] == 'Portland'), 'state_postal_code'] = 'OR'
    dataframe.loc[(dataframe['city'] == 'Denver'), 'state_postal_code'] = 'CO'
    dataframe.loc[(dataframe['city'] == 'Rhode Island'), 'state_postal_code'] = 'RI'
    dataframe.loc[(dataframe['city'] == 'Boston'), 'state_postal_code'] = 'MA'
    dataframe.loc[(dataframe['city'] == 'Oakland'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'San Mateo County'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'Jersey City'), 'state_postal_code'] = 'NJ'
    dataframe.loc[(dataframe['city'] == 'Asheville'), 'state_postal_code'] = 'NC'
    dataframe.loc[(dataframe['city'] == 'Santa Cruz County'), 'state_postal_code'] = 'CA'
    dataframe.loc[(dataframe['city'] == 'Columbus'), 'state_postal_code'] = 'OH'
    dataframe.loc[(dataframe['city'] == 'Cambridge'), 'state_postal_code'] = 'OH'
    dataframe.loc[(dataframe['city'] == 'Salem'), 'state_postal_code'] = 'MA'
    dataframe.loc[(dataframe['city'] == 'Pacific Grove'), 'state_postal_code'] = 'CA'

    list1 = dataframe["state_postal_code"].value_counts().index
    list1e = list1.tolist()

    list2 = ["California", "New York", "Hawaii", "Washington", "Florida", "Texas", "Nevada", "Minnesota", "Louisiana",
             "Illinois", "Tennessee", "Oregon", "Colorado", "Rhode Island", "Massachusetts",
             "New Jersey", "Ohio", "North Carolina"]

    dicts = {k: v for k, v in zip(list1e, list2)}

    for state, name in dicts.items():
        dataframe.loc[(dataframe['state_postal_code'] == state), 'state_name'] = name


    # k means ile segment oluşturma
    col_name = ["calculated_host_listings_count"]
    rfmkmeans = dataframe[col_name]

    sc = MinMaxScaler((0, 1))
    x_sc = sc.fit_transform(rfmkmeans)
    rfm_sc = pd.DataFrame(x_sc)

    # Determining optimum cluster number
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(rfm_sc)

    # results of how many cluster should be
    elbow.elbow_value_

    # Creating final clusters
    kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm_sc)
    kumeler = kmeans.labels_

    pd.DataFrame({"id": rfmkmeans.index, "Segments": kumeler})
    dataframe["cluster_no"] = kumeler


    nltk.download('wordnet')
    dataframe["name"] = dataframe["name"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    comment_words = ''
    stopwords = set(STOPWORDS)
    for val in dataframe.name:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens) + " "

    dataframe["unigrams"] = dataframe["name"].apply(nltk.word_tokenize)
    from nltk.corpus import stopwords
    dataframe["unigrams"] = dataframe["unigrams"].apply(lambda x: [item for item in x if item not in stopwords.words('english')])

    result = pd.Series(np.concatenate([x for x in dataframe.unigrams])).value_counts()
    result = pd.DataFrame({'ngrams': list(result.keys()),'count': list(result[:])})

    result = result[result.ngrams != '.']
    result = result.head(100)
    result1 = result['ngrams'][1:100]
    result1


    def inner(text):
        for word in text.split(' '):
            for c in result1:
                if word == c:
                    return c
        return 'Null'

    c1 = []
    for text in dataframe['name']:
        c1.append(inner(text))
    dataframe['Name1'] = c1


    """ --------------------DROPPING-------------------- """
    # Drop columns have no meaning now after feature engineering, because new features also carrying informations
    drop_list = ["name", "host_name", "id", "host_id", "last_review", "neighbourhood_group", "neighbourhood",
                 "latitude", "longitude", "last_review","city", "state_name", "reviews_per_month",
                 "unigrams"]

    dataframe.drop(drop_list, inplace=True, axis=1)

    """ --------------------ENCODING-------------------- """
    # Label Encoder
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    labeller = ["room_type", "day_name"]

    for col in labeller:
        dataframe = label_encoder(dataframe, col)

    # One-Hot Encoding
    ohe_cols = ["state_postal_code", "Name1", "cluster_no"]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    """ --------------------SCALER-------------------- """
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    numerical_cols = [col for col in num_cols if "price" not in col]


    ms = MinMaxScaler()
    dataframe[numerical_cols] = ms.fit_transform(dataframe[numerical_cols])


    """ --------------------MODELLING-------------------- """
    "assignment of target and independent variables"
    y = dataframe["pricelog_"]
    X = dataframe.drop(["price", "pricelog_"], axis=1)

    return X,y

X, y = airbnb_data_prep(df)

def load_airbnb():
    data = pd.read_csv("datasets/AB_US_2020.csv")
    return data

df = load_airbnb()

X,y = airbnb_data_prep(df)











