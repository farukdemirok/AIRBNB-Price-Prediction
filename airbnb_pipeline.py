################################################
# End-to-End Airbnb Machine Learning Pipeline II
################################################

# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning

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

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime as dt
import pickle
import sys
import warnings
import re
import pandas_profiling
import joblib

if not sys.warnoptions:
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

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

def filling_with_missing(dataframe, col_name, date=False):
    if date:
        dataframe[col_name] = dataframe[col_name].fillna("01/01/1990")

    else:
        dataframe[col_name] = dataframe[col_name].fillna("missing")

def inner(text):
    for word in text.split(' '):
        for c in result1:
            if word == c:
                return c
    return 'Null'

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

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


######################################################
# 3. Base Models
######################################################

def base_models(X, y, scoring="f1"):
    print("Base Models....")
    models = [('RF', RandomForestRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())
          ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=3, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")



######################################################
# Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [5, 8, 10],
             "max_features": [7,8, 9],
             "min_samples_split": [17,18,20,22],
             "n_estimators": [125,150,175]}

xgboost_params = {"learning_rate": [0.2, 0.15, 0.05],
                  "max_depth": [7, 8,9],
                  "n_estimators": [275, 300,325],
                  "colsample_bytree": [0.5,0.6,0.7]}

lightgbm_params = {"learning_rate": [0.01, 0.50, 0.05],
                   "n_estimators": [50,100,150,200],
                   "colsample_bytree": [0.2,0.3,0.4,0.5]}

regressors = [("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]


def hyperparameter_optimization(X,y):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=3, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=3, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models


######################################################
# # Stacking & Ensemble Learning
######################################################

def voting_regressor(best_models, X, y):
    print("Voting Regressor")
    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])])

    voting_reg.fit(X, y)
    np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=3, scoring="neg_mean_squared_error")))
    return voting_reg



################################################
# Pipeline Main Function
################################################
import os

def main():
    df = pd.read_csv(r"C:\Users\fdemr\Desktop\DSMLBC6\datasets\AB_US_2020.csv")
    X, y = airbnb_data_prep(df)
    print("base models...")
    base_models(X, y)
    print("hyperparameter optimization...")
    best_models = hyperparameter_optimization(X, y)
    print("creating best model with voting regressor")
    voting_clf = voting_regressor(best_models, X, y)
    os.chdir(r"C:\Users\fdemr\Desktop\DSMLBC6")
    joblib.dump(voting_clf, "voting_clf_airbnb.pkl")
    print("Voting_clf has been created")
    return voting_clf

if __name__ == "__main__":
    main()



