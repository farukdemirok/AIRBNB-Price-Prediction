################################################
# End-to-End Airbnb Learning Pipeline III
################################################

# 6. Prediction for new observation

import joblib
import pandas as pd


df = pd.read_csv("datasets/AB_US_2020.csv")


random_user = df.sample(1, random_state=45)
new_model = joblib.load("Project/voting_clf_airbnb.pkl")

new_model.predict(random_user)


from Project.airbnb_pipeline import *

X, y = airbnb_data_prep(df)

random_user = X.sample(1, random_state=45)
new_model = joblib.load("Project/voting_clf_airbnb.pkl")
new_model.predict(random_user)



