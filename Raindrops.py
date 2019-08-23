#%%
# import tools and packages

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import lxml.html as lh
import chromedriver_binary


import re
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import seaborn as sns
import matplotlib.pyplot as plt


import statsmodels.api as sm
from sklearn.linear_model import lars_path
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#%%
# Webscrapping
# Go to NEA site
url = "http://www.weather.gov.sg/climate-historical-daily/"
driver = webdriver.Chrome()
driver.get(url)

# Choose Newton
driver.find_element_by_xpath("//button[@id='cityname']").click()

elem_city = driver.find_element_by_xpath(
    "//ul[@class='dropdown-menu long-dropdown']")
for option_city in elem_city.find_elements_by_tag_name('li'):
    # print(option_city)
    if option_city.text == 'Newton':
        option_city.click()
        
# initiate a dataframe for the output
final_df = pd.DataFrame()

# Loop over all the Year options
driver.find_element_by_xpath("//button[@id='year']").click()
elem_year = driver.find_element_by_xpath("//*[@id='yearDiv']/ul")

for option_year in elem_year.find_elements_by_tag_name('li'):
    try:
        driver.find_element_by_xpath("//button[@id='year']").click()
        time.sleep(1)
        option_year.click()
        time.sleep(2)
    except:
        print(option_year)
        driver.find_element_by_xpath("//button[@id='year']").click()
        option_year.click()
        time.sleep(2)


# Loop over all the Year options
    driver.find_element_by_xpath("//button[@id='month']").click()
    elem_month = driver.find_element_by_xpath("//*[@id='monthDiv']/ul")
    print(elem_month.find_elements_by_tag_name('li'))
    for option_month in elem_month.find_elements_by_tag_name('li'):
        # print(option_month.text)
        try:
            driver.find_element_by_xpath("//button[@id='month']").click()
            option_month.click()
        except Exception as e:
            print(e)


# Hit Display and wait
        driver.find_element_by_xpath("//input[@id='display']").click()
        driver.implicitly_wait(5)

# Get the HTML table into pandas to become a dataframe
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        allTables = soup.find_all('table')[0]
        df = pd.read_html(allTables.prettify())[0]

# Concatenate the data
        final_df = pd.concat([final_df, df], axis=0, sort=False)
        print(df)
        print(type(df))
        time.sleep(3)
        
# Close the website
driver.close()


#%%
# Exploratory Data Analysis
final_df.info()

#Reset the index to run sequentially
final_df = final_df.reset_index()

# Duplicate the df and strip column spaces
result1= final_df.copy()
result1.rename(columns=lambda x: x.strip())

# Convert the columns of objects to floats and check only date is object
result1.columns
result1[['Daily Rainfall Total (mm)', 'Highest 30-min Rainfall (mm)','Highest 60-min Rainfall (mm)',\
         'Highest 120-min Rainfall (mm)','Mean Temperature (°C)', 'Maximum Temperature (°C)','Minimum Temperature (°C)',\
         'Mean Wind Speed (km/h)','Max Wind Speed (km/h)']] = \
result1[['Daily Rainfall Total (mm)', 'Highest 30-min Rainfall (mm)','Highest 60-min Rainfall (mm)',\
         'Highest 120-min Rainfall (mm)','Mean Temperature (°C)', 'Maximum Temperature (°C)','Minimum Temperature (°C)', \
         'Mean Wind Speed (km/h)','Max Wind Speed (km/h)']].apply(pd.to_numeric,downcast = 'float', errors='coerce')

result1.info()
# Save results to csv
result1.head()

#%%
# Drop the old index column of sets of 0-30 which was picked up from every table
result2 = result1.drop('index', axis=1)  
result2.head(45)
result2.to_csv("full_output.csv", index = False)

# Data cleansing
# Drop first 30 rows for Changi data to get only Newton data
result3 = result2.drop(result1.head(30).index, axis=0)
result3.head(40)

# Reset the index
result4 = result3.reset_index(drop=True)
result4.head(34)

# Remove rows with missing data
# Data prior to 2014 does not have full rainfall data and\
# there were days between 2014 - 2019 with some empty cells.
result5 = result4.dropna(). reset_index()
result5.info()
# Drop the index since it is no longer sequential
Newton2014 = result5.drop('index', axis=1)
# Export to csv
Newton2014.to_csv("Newton2014.csv", index = False)
# Tally Newton2014 data with results5
Newton2014.info()
Newton2014.shape
Newton2014 = pd.read_csv("Newton2014.csv")


#%%
# Make a copy of Newton2014 data
Newton1419full=Newton2014.copy()
Newton1419full.head()
Newton1419full.tail()

# Rename the columns
Newton1419full.set_axis(['Date','Daily_Rainfall(mm)','Highest30_min_Rainfall','Highest60_min_Rainfall',\
                         'Highest120_min_Rainfall','Mean_Temperature','Max_Temperature','Min_Temperature',\
                         'Mean_Wind_Speed','Max_Wind_Speed'], axis=1, inplace=True)


# configure the format of the images:
%config InlineBackend.figure_format = 'svg'
#allows the visuals to render within Jupyter Notebook
%matplotlib inline 
%pylab inline
# use heatmap to see correlation
sns.heatmap(Newton1419full.corr(), cmap="YlGnBu", annot=True, vmin=-1, vmax=1);
plt.xticks(rotation=87)
plt.savefig("correlation.svg")

# use pairplot to visualize relationships with target and distribution
sns.pairplot(Newton1419full)
plt.savefig("pairplot.svg")

#%%
# Obtain info on Newton full info
Newton1419full.describe()
Newton1419full.columns

# Remove columns that contributes to rainfall to prevent data leakage
Newton_wind_temp= Newton1419full.drop(['Highest30_min_Rainfall', "Highest60_min_Rainfall","Highest120_min_Rainfall"], \
                                  axis = 1)
Newton_wind_temp.head()
#%%
# Model selection and feature selection
# 1. OLS Model
# Noted from earlier pairplot that the remaining parameters do not have obvious patterns
# to do the usual feature engineering, namely functions of x**2 or log 
# Slice data into features and target
X = Newton_wind_temp.drop(columns = ["Daily_Rainfall(mm)", "Date"]).astype(float)
y = Newton_wind_temp.loc[:,"Daily_Rainfall(mm)"].astype(float)

# fit model with Daily_rainfall as target 
Daily_rainfall = sm.OLS(y, X, data=Newton_wind_temp)
Daily_rainfall_analysis = Daily_rainfall.fit()

# summarize OLS Regression model
Daily_rainfall_analysis.summary()

#%%
# Linear Regression, Ridge or Poly
# Split the data into 3 portions: 60% for training, 20% for validation (used to select the model), 20% for final testing evaluation.

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=.25, random_state=42)

#%%
# Features selection with LASSO
# Scale the variables
std = StandardScaler()
std.fit(X_train.values)
X_tr = std.transform(X_train.values)

# Finding the lars paths
print("Computing regularization path using the LARS ...")
alphas, _, coefs = lars_path(X_tr, y_train.values, method='lasso')

# plotting the LARS path
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(8,8))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.legend(X_train.columns)
plt.show()
plt.savefig("LASSO_orig_wind_temp.svg")

#%%
#  Features selection with Ridge
## Scale the variables
std = StandardScaler()
std.fit(X_train.values)
X_tr = std.transform(X_train.values)

# Finding the lars paths
print("Computing regularization path using the LARS ...")
alphas, _, coefs = lars_path(X_tr, y_train.values, method='ridge')

# plotting the LARS path
xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=(8,8))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('Ridge Path')
plt.axis('tight')
plt.legend(X_train.columns)
plt.show()
#plt.savefig("Ridge_orig_wind_temp.svg")


#%%
#set up the 4 models for train & validation & test:
#Ridge alpha = 1 and Lasso alpha = 10
lm = LinearRegression()
lm.fit(X_train, y_train)
print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')
print(f'Linear Regression test R^2: {lm.score(X_test, y_test):.3f}')
#Feature scaling for train/val data so that we can run our ridge/lasso model on each
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)
X_scaled = scaler.transform(X.values)
lm_ridge = Ridge(alpha=1)
lm_ridge.fit(X_train_scaled, y_train)
print(f'Ridge Regression val R^2: {lm_ridge.score(X_val_scaled, y_val):.3f}')
print(f'Ridge Regression test R^2: {lm_ridge.score(X_test_scaled, y_test):.3f}')
lm_lasso = Lasso(alpha=10)
lm_lasso.fit(X_train_scaled, y_train)
print(f'Lasso Regression val R^2: {lm_lasso.score(X_val_scaled, y_val):.3f}')
print(f'Lasso Regression test R^2: {lm_lasso.score(X_test_scaled, y_test):.3f}')
#Feature transforms for train/val data so that we can run our poly model on each
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)
X_poly = poly.transform(X.values)
lm_poly = LinearRegression()
lm_poly.fit(X_train_poly, y_train)
print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
print(f'Degree 2 polynomial regression test R^2: {lm_poly.score(X_test_poly, y_test):.3f}') 


# # cross validation using KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state = 71)
cvs_lm = cross_val_score(lm, X_test, y_test, cv=kf, scoring='r2')
print(("Linear Regression test R^:"),( cvs_lm))
print(("Linear Regression test mean R^:"), round(np.mean(cvs_lm),3), "+-", round(np.std(cvs_lm),3) )
cvs_ridge = cross_val_score(lm_ridge, X_test_scaled, y_test, cv=kf, scoring='r2')
print(('Ridge Regression test R^2:'), cvs_ridge)
print( ("Ridge Regression test mean R^:"), round(np.mean(cvs_ridge),3), "+-", round(np.std(cvs_ridge),3) )
cvs_lasso = cross_val_score(lm_lasso, X_test_scaled, y_test, cv=kf, scoring='r2')
print(('Lasso Regression test R^2:'),cvs_lasso)
print( ("Lasso Regression test mean R^:"), round(np.mean(cvs_lasso),3), "+-", round(np.std(cvs_lasso),3) )
cvs_poly = cross_val_score(lm_poly, X_test_poly, y_test, cv=kf, scoring='r2')
print(('Degree 2 polynomial regression test R^2:'),cvs_poly)
print( ('Degree 2 polynomial regression test mean R^2:'), round(np.mean(cvs_poly),3), "+-", round(np.std(cvs_poly),3) )

#%%
#Ridge alpha = 100 and Lasso alpha = 100
#set up the 4 models for train & validation & test:
lm = LinearRegression()
lm.fit(X_train, y_train)
print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')
print(f'Linear Regression test R^2: {lm.score(X_test, y_test):.3f}')
#Feature scaling for train/val data so that we can run our ridge/lasso model on each
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_val_scaled = scaler.transform(X_val.values)
X_test_scaled = scaler.transform(X_test.values)
X_scaled = scaler.transform(X.values)
lm_ridge = Ridge(alpha=100)
lm_ridge.fit(X_train_scaled, y_train)
print(f'Ridge Regression val R^2: {lm_ridge.score(X_val_scaled, y_val):.3f}')
print(f'Ridge Regression test R^2: {lm_ridge.score(X_test_scaled, y_test):.3f}')
lm_lasso = Lasso(alpha=100)
lm_lasso.fit(X_train_scaled, y_train)
print(f'Lasso Regression val R^2: {lm_lasso.score(X_val_scaled, y_val):.3f}')
print(f'Lasso Regression test R^2: {lm_lasso.score(X_test_scaled, y_test):.3f}')
#Feature transforms for train/val data so that we can run our poly model on each
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)
X_poly = poly.transform(X.values)
lm_poly = LinearRegression()
lm_poly.fit(X_train_poly, y_train)
print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')
print(f'Degree 2 polynomial regression test R^2: {lm_poly.score(X_test_poly, y_test):.3f}') 


# # cross validation using KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state = 71)
cvs_lm = cross_val_score(lm, X_test, y_test, cv=kf, scoring='r2')
print(("Linear Regression test R^:"),( cvs_lm))
print(("Linear Regression test mean R^:"), round(np.mean(cvs_lm),3), "+-", round(np.std(cvs_lm),3) )
cvs_ridge = cross_val_score(lm_ridge, X_test_scaled, y_test, cv=kf, scoring='r2')
print(('Ridge Regression test R^2:'), cvs_ridge)
print( ("Ridge Regression test mean R^:"), round(np.mean(cvs_ridge),3), "+-", round(np.std(cvs_ridge),3) )
cvs_lasso = cross_val_score(lm_lasso, X_test_scaled, y_test, cv=kf, scoring='r2')
print(('Lasso Regression test R^2:'),cvs_lasso)
print( ("Lasso Regression test mean R^:"), round(np.mean(cvs_lasso),3), "+-", round(np.std(cvs_lasso),3) )
cvs_poly = cross_val_score(lm_poly, X_test_poly, y_test, cv=kf, scoring='r2')
print(('Degree 2 polynomial regression test R^2:'),cvs_poly)
print( ('Degree 2 polynomial regression test mean R^2:'), round(np.mean(cvs_poly),3), "+-", round(np.std(cvs_poly),3) )


#%%
# In view of colinearity of Temperature elements, select: 
# only 1 Temperature element (Min Temp) and the 2 (not-overly-colinear) Wind elements

Newton1419 = Newton1419full.drop(columns=['Highest30_min_Rainfall',
       'Highest60_min_Rainfall', 'Highest120_min_Rainfall', 'Mean_Temperature',
       'Max_Temperature'])

# slice data into features and target
X = Newton1419.drop(columns = ["Daily_Rainfall(mm)", "Date"]).astype(float)
y = Newton1419.loc[:,"Daily_Rainfall(mm)"].astype(float)

# fit model with target as Daily_rainfall
Daily_rainfall = sm.OLS(y, X, data=Newton1419)

Daily_rainfall_analysis = Daily_rainfall.fit()

# summarize our model
Daily_rainfall_analysis.summary()

#%%
# Conclusion: OLS with all the Temp and Wind Elements have the best adjusted R^2 and p values
# Back to full temperature and wind OLS X and y
X = Newton_wind_temp.drop(columns = ["Daily_Rainfall(mm)", "Date"]).astype(float)
y = Newton_wind_temp.loc[:,"Daily_Rainfall(mm)"].astype(float)

# develop OLS with Sklearn 
lr = LinearRegression()
fit = lr.fit(X,y) # for later use
# Plot your predicted values on the x-axis, and your residuals on the y-axis

Newton_wind_temp['predict']=fit.predict(X)
Newton_wind_temp['resid']=y-Newton_wind_temp.predict

with sns.axes_style('white'):
    plot=Newton_wind_temp.plot(kind='scatter',
                  x='predict',y='resid',alpha=0.2,figsize=(10,6))

plt.show()

# inspect histogram
Newton_wind_temp['Daily_Rainfall(mm)'].hist(bins=30)
plt.title('Histogram of Dependent Variable')
plt.show()

# diagnose/inspect residual normality using qqplot:
stats.probplot(Newton_wind_temp['resid'], dist="norm", plot=plt)
plt.figure(figsize=(8,8))

plt.show()

#%%
# inspect sqrt histogram
np.sqrt(Newton_wind_temp['Daily_Rainfall(mm)']).hist(bins=30)
plt.title('Histogram of Dependent Variable')
plt.show()


# inspect log histogram
np.log(Newton_wind_temp['Daily_Rainfall(mm)']+1).hist(bins=30)
plt.title('Histogram of Dependent Variable')
plt.show()

# inspect to examine for box cox transformation
fig = plt.figure()
ax = fig.add_subplot(111)
prob = stats.boxcox_normplot(Newton_wind_temp['Daily_Rainfall(mm)']+1, 1, 3, plot=ax)
#%%
