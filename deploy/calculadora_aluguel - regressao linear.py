#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.regressor import ResidualsPlot


# # Reading data

# In[2]:


df = pd.read_csv("../properties_v2.csv", sep=",")


# In[3]:


df['neighbourhood'] = df['neighbourhood'].astype(str)


# In[4]:


bool_cols = [col for col in df if 
               df[col].dropna().value_counts().index.isin([0,1]).all()]


# In[5]:


# numerical_features = ["area", "rooms", "bathrooms", "garages"]
numerical_features = ["area", "bathrooms", "garages"]
categorical_features = ["neighbourhood"]
binary_features = bool_cols
target = "price"


# In[6]:


# df.isnull().sum()[:10]


# Condominio muito missing:
# Opcao 1a: Remover a coluna
# Opcao 1b: Remover as observacoes faltantes
# Opcao 2: Imputation (completar os valores faltantes)
# Opcao 3: Usar mesmo assim

# In[7]:


df = df.dropna(subset = ["area", "rooms", "bathrooms", "garages", "price"])


# # Data pipeline

# In[8]:


# numerical_features


# In[9]:


scaler = MinMaxScaler()
data_pipeline = ColumnTransformer([("numerical", scaler, numerical_features)], 
                                  remainder="passthrough")


# In[10]:


X = df[numerical_features + categorical_features + binary_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[11]:


data_pipeline.fit(X_train)


# In[12]:


X_train_transformed = data_pipeline.transform(X_train)
X_test_transformed = data_pipeline.transform(X_test)


# In[13]:


X_train_transformed = pd.DataFrame(X_train_transformed, columns=numerical_features+categorical_features+binary_features)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=numerical_features+categorical_features+binary_features)


# # Feature selection

# In[14]:


selector = SelectKBest(score_func=mutual_info_regression, k="all")
selector.fit(X_train_transformed[numerical_features+binary_features], y_train)


# In[15]:


pd.DataFrame(zip(numerical_features + binary_features, selector.scores_), columns=["feature", "score"]).sort_values("score", ascending=False).head(10)


# # Fitting model

# # Catboost

# In[16]:


scaler = MinMaxScaler()
data_pipeline = ColumnTransformer([("numerical", scaler, numerical_features)], 
                                  remainder="passthrough")


# In[17]:


X = df[numerical_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[18]:


data_pipeline.fit(X_train)


# In[19]:


X_train_transformed = data_pipeline.transform(X_train)
X_test_transformed = data_pipeline.transform(X_test)


# In[20]:


X_train_transformed = pd.DataFrame(X_train_transformed, columns=numerical_features+categorical_features)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=numerical_features+categorical_features)


# In[21]:


model = CatBoostRegressor(cat_features=["neighbourhood"])
model.fit(X_train_transformed, y_train, eval_set=(X_test_transformed, y_test), verbose=False)


# In[22]:


y_pred = model.predict(X_test_transformed)
y_pred_train = model.predict(X_train_transformed)


# In[23]:


mape_catboost = mean_absolute_percentage_error(y_test, y_pred)


# In[24]:


r2_catboost = r2_score(y_test, y_pred)


# In[25]:


mape_catboost_train = mean_absolute_percentage_error(y_train, y_pred_train)


# In[26]:


r2_catboost_train = r2_score(y_train, y_pred_train)


# In[27]:


wrapped_model = wrap(model)
visualizer = ResidualsPlot(wrapped_model)

visualizer.fit(X_train_transformed, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test_transformed, y_test)  # Evaluate the model on the test data
# visualizer.show()         

