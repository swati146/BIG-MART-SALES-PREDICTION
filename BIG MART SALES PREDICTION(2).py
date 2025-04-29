#!/usr/bin/env python
# coding: utf-8

# # Big Mart Sales Prediction
# 1. Introduction
# The Big Mart Sales Prediction project aims to forecast sales for products across various Big Mart outlets. By using machine learning algorithms, we predict Item Outlet Sales based on item and outlet attributes. Accurate sales forecasting helps Big Mart manage inventory, plan marketing strategies, and improve supply chain operations.
# 2. Dataset Overview
# The project utilizes two datasets: Train.csv (with sales numbers) and Test.csv (without sales numbers). Key features include Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type, Outlet_Type, and Item_Outlet_Sales (target variable).
# 

# In[1]:


import numpy as np # 
import pandas as pd #
import math
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from xgboost import XGBRegressor
import optuna

# Ignore warnings ;)
import warnings
warnings.simplefilter("ignore")

import pickle

# set seed for reproductibility
np.random.seed(0)


# In[2]:


# loading the data 
train = pd.read_csv(r"C:/Users/Dileep/Downloads/train_v9rqX0R.csv")


# In[3]:


train


# In[4]:


train.Item_Type.value_counts()


# In[5]:


# train_columns=train[['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']]


# In[6]:


train.info()


# In[7]:


train.describe(include='all')


#  Some columns like Item_Weight and Outlet_Size have missing values (not complete).
# 
# Categorical features like Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Size, Outlet_Location_Type, and Outlet_Type have a few distinct categories (e.g., 5 types of Item_Fat_Content).
# 
# Numerical features like Item_Visibility, Item_MRP, and Item_Outlet_Sales show large ranges, indicating diverse sales and pricing.
# 
# The average Item_Weight is about 12.86 kg, and the average Item_MRP is about 141 currency units.
# 
# Outlet_Establishment_Year ranges from 1985 to 2009, meaning the outlets vary significantly in age.
# 
# Sales (Item_Outlet_Sales) vary widely, from 33 to 13,087 units, suggesting high variability in product success.

# In[8]:


train.isnull().sum()##checking the null values in the dataframe


# # DATA PREPROCCESSING

# ## Handling the missing values

# In[9]:


train['Item_Weight'] = train['Item_Weight'].fillna(train.groupby('Item_Type')['Item_Weight'].transform('mean'))


# In[10]:


train['Item_Weight']


# In[11]:


train['Outlet_Size'] = train['Outlet_Size'].fillna(
    train.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0])
)



#  For 'Item_Weight':
# 
# Missing values are filled with the mean weight of the respective 'Item_Type' group.
# 
# This ensures that items of the same type get their missing weight imputed with an average value that represents that specific item category, making the imputation more relevant.
# 
# 
# For 'Outlet_Size':
# 
# Missing values are filled with the mode (most frequent value) of the respective 'Outlet_Type' group.
# 
# This is because certain outlet types (like 'Supermarket Type1' or 'Grocery Store') might have a more common size (e.g., Medium or Small), and filling with the mode preserves the category's typical characteristic.
# 
# This approach customizes the filling by considering relationships within the data (e.g., item types and outlet types), which ensures that missing values are filled in a way that aligns better with the overall structure of the dataset.

# In[12]:


train.isnull().sum()


# In[13]:


mean_sales = train['Item_Outlet_Sales'].mean()


# In[14]:


mean_sales


# In[15]:


train['Item_Fat_Content'].unique()


#  standardizes the 'Item_Fat_Content' column by first converting all text to lowercase to avoid case mismatches. Then, it maps different variations (like 'low fat', 'lf', 'regular', and 'reg') to their correct, consistent categories ('Low Fat' and 'Regular'). This ensures uniformity in the values for better analysis and modeling.

# In[16]:


# First, make everything lowercase to avoid case mismatch
train['Item_Fat_Content'] = train['Item_Fat_Content'].str.lower()

# Now map all variations to correct categories
fat_content_mapping = {
    'low fat': 'Low Fat',
    'lf': 'Low Fat',
    'regular': 'Regular',
    'reg': 'Regular'
}

train['Item_Fat_Content'] = train['Item_Fat_Content'].map(fat_content_mapping)





# In[17]:


train['Item_Fat_Content'].unique()


# In[18]:


train['Item_Visibility']


# We replace zero values in 'Item_Visibility' with the mean visibility to correct potential data errors or missing values. This ensures the model doesn't misinterpret zeros as valid values, improving overall accuracy.

# In[19]:


# Create median visibility per Item_Type
visibility_median_per_type = train.groupby('Item_Type')['Item_Visibility'].transform('median')

# Replace 0 with the median of their Item_Type
train['Item_Visibility'] = train.apply(
    lambda x: visibility_median_per_type[x.name] if x['Item_Visibility'] == 0 else x['Item_Visibility'],
    axis=1
)




# In[20]:


train['Item_Visibility']


# # Data exploration

# In[21]:


numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols


# In[22]:


train.describe().T


# Item_Weight: The average weight is 12.86 with values ranging from 4.56 to 21.35.
# 
# Item_Visibility: The average visibility is 0.0661, with values ranging from 0 to 0.3284, suggesting some items may have zero visibility (which could be a potential issue).
# 
# Item_MRP: The average maximum retail price (MRP) is 140.99, with a significant spread from 31.29 to 266.89.
# 
# Outlet_Establishment_Year: Most outlets are established around 1997, with the range from 1985 to 2009.
# 
# Item_Outlet_Sales: The average sales are 2181.29, with a wide spread, from 33.29 to 13086.96, indicating the

# In[23]:


# 2. Descriptive Statistics
# Mean, median, min, max of Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales.

# Distributions of numerical columns (histograms).

# See if anything looks skewed.


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year', 'Item_Outlet_Sales']

plt.figure(figsize=(16, 10))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(16, 8))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=train[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


# ### Summary 
# 
# ![image.png](attachment:image.png)
# 
# 

# In[26]:


# 3. Sales Analysis
# Top 5 selling item types based on total sales (Item_Type vs Item_Outlet_Sales).

# Which Outlet sells the most? (Outlet_Identifier vs Item_Outlet_Sales).

# Impact of MRP on sales (high MRP = more/less sales?).

# Sales across different Outlet Types (Supermarket Type1, Grocery Store, etc.)


# In[27]:


top_5_items = train.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False).head(5)

print(top_5_items)


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(x=top_5_items.values, y=top_5_items.index, palette='viridis')
plt.title('Top 5 Selling Item Types Based on Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Item Type')
plt.show()


# In[29]:


# Group by Outlet_Identifier and sum Item_Outlet_Sales
outlet_sales = train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()

# Sort in descending order
top_outlet = outlet_sales.sort_values(ascending=False)

# Display the result
print(top_outlet)


# In[30]:


plt.figure(figsize=(10,6))
sns.barplot(x=top_outlet.values, y=top_outlet.index, palette="mako")
plt.title('Total Sales by Outlet')
plt.xlabel('Total Sales')
plt.ylabel('Outlet Identifier')
plt.show()


# ### Insights:		
# ![image.png](attachment:image.png)
# 

# In[31]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Item_MRP', y='Item_Outlet_Sales', data=train, alpha=0.5)
plt.title('Impact of MRP on Sales')
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.show()


# ### Insights:	
# Key Observations:
# There is a positive relationship between Item_MRP and Item_Outlet_Sales: higher MRP items generally have higher sales.
# 
# However, the plot is clearly segmented into 4 distinct price bands:
# 
# Around 0–70, 70–130, 130–200, and 200–270.
# 
# Within each MRP band, sales still vary widely — not all expensive items always sell more, but the potential for higher sales increases with MRP.
# 
# Clustered sales at lower levels indicate that many low and mid-priced items are sold in large quantities too.
# ![image.png](attachment:image.png)
# 
# 
# Business Implications:
# Premium products (high MRP) can drive higher revenues but need marketing because not every expensive item sells automatically.
# 
# Mid-range products might be the sweet spot for volume sales.
# 
# Price segmentation strategy seems already in place, but exploring bundling, promotions, or loyalty programs within each price band could boost overall sales.
# 

# In[32]:


# Correlation between MRP and Sales
correlation = train['Item_MRP'].corr(train['Item_Outlet_Sales'])
print(f"Correlation between MRP and Sales: {correlation:.2f}")


# The correlation between MRP (Maximum Retail Price) and Item_Outlet_Sales is 0.57, which indicates a moderate positive relationship between the two variables. Here's the interpretation:
# 
# Moderate Positive Correlation: A correlation of 0.57 suggests that as the MRP of an item increases, the sales tend to increase as well, but not perfectly. This means that higher-priced items tend to have higher sales, but the relationship isn't perfectly linear. Other factors likely influence sales beyond just the price.
# 
# Potential Implications:
# 
# Higher-priced items (MRP) may attract more attention or be considered premium products, leading to higher sales.
# 
# However, this isn't a very strong correlation, so other factors (like outlet size, location, marketing, or consumer preferences) also play a significant role in determining sales.
# 
# Sales Strategy: For businesses, this could mean that pricing strategy (such as setting higher MRPs for certain products) might positively impact sales, but other variables should be considered for more accurate predictions or strategies.
# 
# In short, while there's a noticeable trend where higher MRPs are somewhat related to higher sales, it’s not a perfect correlation, so other factors are influencing sales as well.

# In[33]:


plt.figure(figsize=(10,6))
sns.barplot(x='Outlet_Type', y='Item_Outlet_Sales', data=train, estimator=sum, ci=None)
plt.title('Total Sales across Different Outlet Types')
plt.xlabel('Outlet Type')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()


# Key Insights:
# 
# Supermarket Type1 outlets are the major revenue contributors.
# 
# Grocery Stores contribute very little to total sales.
# 
# It suggests that outlet type strongly affects sales — larger or more modern supermarkets (like Type1) likely attract more customers and generate more sales compared to smaller or traditional stores like Grocery Stores.

# In[34]:


train['Item_Outlet_Sales'] = np.log1p(train['Item_Outlet_Sales'])




# In[35]:


train['Item_Outlet_Sales']


# In[36]:


mean_sales_after_log = train['Item_Outlet_Sales'].mean()


# In[37]:


mean_sales_after_log


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(data=train, x='Item_MRP', y='Item_Outlet_Sales')
plt.title('Item MRP vs Item Outlet Sales')
plt.xlabel('Item MRP')
plt.ylabel('Item Outlet Sales')
plt.show()


# There is a positive trend — as Item MRP increases, Item Outlet Sales also tend to increase.
# 
# The data forms four distinct bands, suggesting that prices (MRP) are grouped into specific ranges or tiers.
# 
# Higher-priced items generally achieve higher sales values, but there’s also variation within each MRP group.

# # Feature Creation

# ### 1) MRP_CATEGORY

# In[39]:


# In training
bin_edges = np.arange(0, train['Item_MRP'].max() + 20, 20)
train['MRP_bins'] = pd.cut(train['Item_MRP'], bins=bin_edges)


# In[40]:


train['MRP_Category'] = pd.cut(train['Item_MRP'],
                               bins=[0, 70, 140, 210, 310],
                               labels=['Low', 'Medium', 'High', 'Very High'])


# Purpose:
# MRP_bins:
# 
# Helps the model detect small variations in price.
# 
# Useful when you want more granular patterns — for example, sales difference between Rs.100 and Rs.120.
# 
#  MRP_Category:
# 
# Simplifies price into broader levels — Low, Medium, High, Very High.
# 
# Useful to reduce noise and capture general pricing trends (e.g., high-priced items may sell less than low-priced).

# ### 2) Visibility Adjustment Feature

# In[41]:



# Replace zeros with mean visibility
visibility_mean = train['Item_Visibility'].mean()
train['Item_Visibility'] = train['Item_Visibility'].replace(0, visibility_mean)

# Create a "Visibility Mean Ratio"
train['Visibility_MeanRatio'] = train['Item_Visibility'] / visibility_mean




# fixing invalid zeros by replacing them with the mean visibility.
# 
# You create a new ratio feature to tell how an item's visibility compares to the overall average, giving the model better signals.
# 
# 

# ### 3) Outlet Age

# In[42]:



current_year = 2025
train['Outlet_Age'] = current_year - train['Outlet_Establishment_Year']



# Created Outlet_Age to tell the model how old each store is, because store age can affect sales patterns.

# In[43]:


train


# # Encoding Categorical Variables
# Categorical features were transformed using label encoding and one-hot encoding to prepare them for machine learning models.
# 

# In[44]:



encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type','Outlet_Size', 'Outlet_Location_Type']

for feature in ordinal_features:
    train[feature] = encoder.fit_transform(train[feature])

train.shape




# In[45]:


cols_to_encode = [col for col in ['Item_Type', 'Outlet_Identifier','MRP_Category'] if col in train.columns]

train = pd.get_dummies(train, columns=cols_to_encode, drop_first=True)




# In[46]:


train


# In[47]:


train.columns.tolist()


# In[48]:


# Let's drop useless columns
cols_to_drop = ['Item_Identifier','MRP_bins']


train.drop(labels=cols_to_drop, axis=1, inplace=True)


# In[49]:


X = train.drop('Item_Outlet_Sales', axis=1)
y = train['Item_Outlet_Sales']


# In[50]:


X


# In[51]:


X.head()


# In[52]:


y.head()


# # Model building

# ### Linear Regressor

# In[53]:


# splitting into training set and test set 80%-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[54]:


lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)


# In[55]:


lin_reg_predictions = lin_reg_model.predict(X_test)


# In[56]:


print('Training score  : {}'.format(lin_reg_model.score(X_train, y_train)))
print('Test score      : {}'.format(lin_reg_model.score(X_test, y_test)))


# ### Random Forest Regressor

# In[57]:


rand_forest_model = RandomForestRegressor(max_depth=5, n_estimators=200, random_state=42)


rand_forest_model.fit(X_train, y_train)


# ###Hyperparameter tuning 

# In[58]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a base model
rand_forest_model = RandomForestRegressor(random_state=42)

# Randomized search
rand_search = RandomizedSearchCV(
    estimator=rand_forest_model,
    param_distributions=param_dist,
    n_iter=50,       # Number of parameter settings sampled
    cv=3,            # Cross-validation folds
    verbose=2,
    random_state=42,
    n_jobs=-1        # Use all processors
)

# Fit
rand_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", rand_search.best_params_)


# In[59]:


from sklearn.ensemble import RandomForestRegressor

# Best model with tuned hyperparameters
best_rand_forest = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features='auto',
    max_depth=5,
    random_state=42
)

# Fit the model
best_rand_forest.fit(X_train, y_train)

# Predict
y_train_pred = best_rand_forest.predict(X_train)
y_test_pred = best_rand_forest.predict(X_test)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error

print("Train R2 Score:", r2_score(y_train, y_train_pred))
print("Test R2 Score:", r2_score(y_test, y_test_pred))
print("Test RMSE:", mean_squared_error(y_test, y_test_pred, squared=False))


# ### LIGHTGBM Regressor

# In[60]:


pip install lightgbm


# In[61]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[62]:


from sklearn.metrics import r2_score, mean_squared_error

# Train the model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Make predictions for both train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R2 Score for Train and Test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate RMSE for Test set
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Print the results
print(f"Train R2 Score: {train_r2}")
print(f"Test R2 Score: {test_r2}")
print(f"Test RMSE: {test_rmse}")


# ### XGBOOST Regressor

# In[63]:


import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

# Train the model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions for both train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R2 Score for Train and Test sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate RMSE for Test set
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# Print the results
print(f"Train R2 Score: {train_r2}")
print(f"Test R2 Score: {test_r2}")
print(f"Test RMSE: {test_rmse}")


# ### Saving the model

# In[64]:


import joblib

model_save=joblib.dump(best_rand_forest, 'RF.pkl')


# # Prediction

# In[65]:


# Testing using test data


# In[66]:


test  = pd.read_csv( r"C:/Users/Dileep/Downloads/test_AbJTz2l.csv")


# In[67]:


test_columns=test[['Item_Identifier','Outlet_Identifier']]


# In[68]:


test['Item_Weight'] = test['Item_Weight'].fillna(test.groupby('Item_Type')['Item_Weight'].transform('mean'))


# In[69]:


test['Outlet_Size'] = test['Outlet_Size'].fillna(
    test.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0])
)


# In[70]:


# Create median visibility per Item_Type
visibility_median_per_type = test.groupby('Item_Type')['Item_Visibility'].transform('median')

# Replace 0 with the median of their Item_Type
test['Item_Visibility'] = test.apply(
    lambda x: visibility_median_per_type[x.name] if x['Item_Visibility'] == 0 else x['Item_Visibility'],
    axis=1
)


# In[71]:



# In test
test['MRP_bins'] = pd.cut(test['Item_MRP'], bins=bin_edges)


# In[72]:


test['MRP_Category'] = pd.cut(test['Item_MRP'],
                               bins=[0, 70, 140, 210, 310],
                               labels=['Low', 'Medium', 'High', 'Very High'])


# In[73]:



# 2. Visibility Adjustment Feature
# Replace zeros with mean visibility
visibility_mean = test['Item_Visibility'].mean()
test['Item_Visibility'] = test['Item_Visibility'].replace(0, visibility_mean)

# Create a "Visibility Mean Ratio"
test['Visibility_MeanRatio'] = test['Item_Visibility'] / visibility_mean


# In[74]:


# 3. Outlet Age
current_year = 2025
test['Outlet_Age'] = current_year - test['Outlet_Establishment_Year']


# In[75]:



encoder = LabelEncoder()
ordinal_features = ['Item_Fat_Content', 'Outlet_Type','Outlet_Size', 'Outlet_Location_Type']

for feature in ordinal_features:
    test[feature] = encoder.fit_transform(test[feature])

test.shape


# In[76]:


cols_to_encode = [col for col in ['Item_Type', 'Outlet_Identifier','MRP_Category'] if col in test.columns]

test = pd.get_dummies(test, columns=cols_to_encode, drop_first=True)


# In[77]:


test


# In[78]:


# Let's drop useless columns
cols_to_drop = ['Item_Identifier','MRP_bins']



test.drop(labels=cols_to_drop, axis=1, inplace=True)


# In[79]:


test.shape


# In[80]:


train.shape


# In[81]:


model_save = joblib.load('RF.pkl')


# In[82]:


predictions = model_save.predict(test)


# In[83]:


predictions


# In[84]:


test_columns['Item_Outlet_Sales'] = predictions


# In[85]:


test_columns['Item_Outlet_Sales'] = np.expm1(test_columns['Item_Outlet_Sales'])


# In[86]:


test_columns.to_csv("solution_main1.csv")


# In[87]:


df=pd.read_csv("solution_main1.csv")


# In[88]:


df


# # shap analysis

# In[89]:


pip install shap


# In[90]:


import shap
# Create an explainer
explainer = shap.TreeExplainer(model_save)  # Use TreeExplainer for tree-based models like RandomForest
shap_values = explainer.shap_values(X_train)


# In[91]:


shap.summary_plot(shap_values, X_train)


# SHAP Summary Plot Insights:
# - Outlet_Type emerged as the most influential feature.
# - Item_MRP also had a strong positive impact, where higher prices often related to higher predicted sales.
# - Outlet_Age and Outlet_Establishment_Year significantly contributed, suggesting outlet maturity influences sales.
# - Specific outlet identifiers like OUT027, OUT018, and OUT019 influenced performance, indicating store-specific behaviors.
# - Item-related factors like weight, visibility ratios, and item categories played moderate roles.
# 
# 6.2 Color Interpretation:
# The color gradient (blue to pink) represents feature value magnitudes. Higher values generally shift predictions up or down depending on the feature’s impact.
# 
# 6.3 Final Interpretation:
# Outlet characteristics dominate predictive strength, followed by product MRP and select visibility features. Understanding feature impacts allows Big Mart to align marketing and operational efforts based on feature importance insights.
# 7. Results and Conclusion
# Extensive feature engineering, thoughtful preprocessing, and robust model evaluation led to Random Forest emerging as the best model. The model, along with SHAP-based insights, provides actionable strategies for inventory planning, targeted promotions, and operational improvements.
# 

# In[ ]:




