

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('kc_house_data.csv')
df.head()
print ("loaddata&ImportLib")
print (df.head())
print("Data Desciption")
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.info())
df.isnull().sum()
df.duplicated().sum()
print("Dataset")
print(df.describe())
print("Distribution of Target")
plt.figure(figsize = (5,3))
sns.histplot(x = 'price', data = df)
plt.title('Distribution of Target')
plt.show()
print("The Distribution of Some Numerical Features")
list1 = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement','yr_built', 'yr_renovated']

plt.figure(figsize = (15,8))
for i, col in enumerate(list1):
    plt.subplot(2,3,i+1)
    sns.histplot(x = col, data = df)
plt.suptitle('Distributions of Some Numerical Features')
plt.show()
plt.figure(figsize = (15,8))
for i, col in enumerate(list1):
    plt.subplot(2,3,i+1)
    sns.scatterplot(x = col, y='price', data = df)
plt.suptitle('Relationships Between Price And Some Numerical Features')
plt.show()
list1_y = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement','yr_built', 'yr_renovated', 'price']
corr = df[list1_y].corr()
print("Pearson correlation between 'price' and features:")
corr.iloc[6,0:6].sort_values(ascending=False)
print("there is strong linear relationship between house size and price")
list2 = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'view']

plt.figure(figsize = (15,8))
for i, col in enumerate(list2):
    plt.subplot(2,3,i+1)
    sns.countplot(x = col, data = df)
plt.suptitle('Distributions of Some Features')
plt.show()
list2 = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'view']

plt.figure(figsize = (15,8))
for i, col in enumerate(list2):
    plt.subplot(2,3,i+1)
    sns.barplot(x = col, y='price', data = df)
plt.suptitle('Relationships Between Price And Some Features')
plt.show()
list2_y = ['bedrooms', 'bathrooms', 'floors', 'condition', 'grade', 'view', 'price']
corr = df[list2_y].corr()
print("Pearson correlation between 'price' and features:")
corr.iloc[6,0:6].sort_values(ascending=False)
df['zipcode'].value_counts()
print("printing ZIP Codes")
plt.figure(figsize = (15,5))
sns.scatterplot(x='zipcode', y='price', data = df)
plt.title('Relationship Between Price And zipcode')
plt.show()
corr = df[['zipcode', 'price']].corr()
print("Pearson correlation between 'price' and ''zipcode':")
corr
print("Conclusion: the Pearson correlation between price and zipcode is weak. But the scatterplot show zipcode can affect price dramatically. The influence of zipcode cannot be revealed by zipcode numbers, which have no meaning. We need do one hot encoding for zipcode.")
print("Data Cleaning and Feature Engineering - Transform Data")


df.drop('id', axis=1, inplace=True)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df.drop("date", axis=1, inplace=True)
data0 = df.copy() # original data
data0.head()
print(data0.head())
print("Transforming skewed variables")
data1 = df.copy()
skew_limit = 0.75 
skew_vals = data1.skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

print("skew_cols")
print(skew_cols)   # list skew variables

col_with_zero = []
col_without_zero = []

for col in skew_cols.index.values:
    if data1[col].gt(0).all().all():
       col_without_zero.append(col)
    else:
      if data1[col].ge(0).all().all():
         col_with_zero.append(col)

print('col_with_zero', col_with_zero)
print('col_without_zero', col_without_zero)
for col in col_without_zero:
    data1[col] = np.log(data1[col])

for col in col_with_zero:
    data1[col] = np.sqrt(data1[col])
skew_vals = data1.skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

skew_cols   # list skew variables
print("One-hot encoding")
data2 = pd.get_dummies(data1, columns=['zipcode'], drop_first=True) 
print(data2) # one hot encoded data
print("Model Training and Testing")
print("Linear Regression")
y_col = "price"
X = data0.drop(y_col, axis=1)
y = data0[y_col]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72018)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

s = StandardScaler()
lr = LinearRegression()
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)
print(r2_score(y_pred, y_test))
print("Lineear Coeeficient")
print(lr.coef_)
feature_importances = pd.DataFrame(zip(X.columns.tolist(), lr.coef_.ravel()))
print(feature_importances.sort_values(by=1))
print("Polynomial Regression")
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2, include_bias=False,)
X_train_pf = pf.fit_transform(X_train)
X_test_pf = pf.transform(X_test)
X_train_pf_s = s.fit_transform(X_train_pf)
X_test_pf_s = s.transform(X_test_pf)
lr.fit(X_train_pf_s, y_train)
y_pred = lr.predict(X_test_pf_s)
r2_score(y_pred, y_test)
print("Ridge Regression")
from sklearn.linear_model import Ridge

alpha_space = np.geomspace(0.0000001, 1000, num=10)
r2_scores = []

for alpha in alpha_space:
    ridge = Ridge(alpha=alpha, max_iter=100000)
    ridge.fit(X_train_pf_s, y_train)
    y_pred = ridge.predict(X_test_pf_s)
    r2_scores.append(r2_score(y_pred, y_test))

print(r2_scores)
print("Data After Skewed Transformation -- Linear Regression")
y_col = "price"
X = data1.drop(y_col, axis=1)
y = data1[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72018)
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)
print(r2_score(y_pred, y_test))
feature_importances = pd.DataFrame(zip(X.columns.tolist(), lr.coef_.ravel()))
print(feature_importances.sort_values(by=1))
print("Polynomial Regression")
X_train_pf = pf.fit_transform(X_train)
X_test_pf = pf.transform(X_test)
X_train_pf_s = s.fit_transform(X_train_pf)
X_test_pf_s = s.transform(X_test_pf)
lr.fit(X_train_pf_s, y_train)
y_pred = lr.predict(X_test_pf_s)
print(r2_score(y_pred, y_test))
print("Ridge Regression")
alpha_space = np.geomspace(0.0000001, 1000, num=10)
r2_scores = []

for alpha in alpha_space:
    ridge = Ridge(alpha=alpha, max_iter=100000)
    ridge.fit(X_train_pf_s, y_train)
    y_pred = ridge.predict(X_test_pf_s)
    r2_scores.append(r2_score(y_pred, y_test))

print(r2_scores)
print("Data After One hot Encoding -- Linear Regression")

y_col = "price"
X = data2.drop(y_col, axis=1)
y = data2[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72018)
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)
print(r2_score(y_pred, y_test))
feature_importances = pd.DataFrame(zip(X.columns.tolist(), lr.coef_.ravel()))
print(feature_importances.sort_values(by=1))
print("Polynomial Regression")
X_train_pf = pf.fit_transform(X_train)
X_test_pf = pf.transform(X_test)
X_train_pf_s = s.fit_transform(X_train_pf)
X_test_pf_s = s.transform(X_test_pf)
lr.fit(X_train_pf_s, y_train)
y_pred = lr.predict(X_test_pf_s)
print(r2_score(y_pred, y_test))

print("Ridge Regression")

alpha_space = np.geomspace(0.0001, 100000, num=20)
r2_scores_1 = []

for alpha in alpha_space:
    ridge = Ridge(alpha=alpha, max_iter=100000)
    ridge.fit(X_train_pf_s, y_train)
    y_pred = ridge.predict(X_test_pf_s)
    r2_scores_1.append(r2_score(y_pred, y_test))

print(r2_scores_1)

print("One hot coded data without unskewed transformation")

data3 = pd.get_dummies(data0, columns=['zipcode'], drop_first=True)
y_col = "price"
X = data3.drop(y_col, axis=1)
y = data3[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72018)
X_train_pf = pf.fit_transform(X_train)
X_test_pf = pf.transform(X_test)
X_train_pf_s = s.fit_transform(X_train_pf)
X_test_pf_s = s.transform(X_test_pf)
alpha_space = np.geomspace(0.0001, 100000, num=20)
r2_scores_2 = []

for alpha in alpha_space:
    ridge = Ridge(alpha=alpha, max_iter=100000)
    ridge.fit(X_train_pf_s, y_train)
    y_pred = ridge.predict(X_test_pf_s)
    r2_scores_2.append(r2_score(y_pred, y_test))

print(r2_scores_2)


plt.figure(figsize = (6,4))
plt.semilogx(alpha_space, r2_scores_1, 'r',label='with skew data transformation')
plt.semilogx(alpha_space, r2_scores_2, 'b',label='without skew data transformation')
plt.xlabel('alpha')
plt.ylabel('r2_score')
plt.xlim([1e-6,1e6])
plt.ylim([0.2,0.95])
plt.title('r2_score of redge regression')
plt.legend()
plt.show()

print ("Final Results")


r2_dict = {'original data':[0.588, 0.799, 0.799], 
           'after skewed transformation':[0.695,0.797,0.797 ], 
           'after skewed transformation and one hot encoding':[0.865, 0, 0.876]}

r2_df = pd.DataFrame.from_dict(r2_dict, orient='index', columns=['Linear','Polynomial','Ridge'])
print(r2_df)

print(" Both skewed transformation and one hot encoding significantly improve the performance of linear regression. (2) Both skewed transformation and one hot encoding reduce the performance of polynomial regression. (3) One hot encoding significantly improve the performance of ridge regression, while skewed transformation does not help. (4) The Best r2_score comes from ridge regression (0.890) for one hot encoding data without skewed transformation. (5) The best r2_score of linear regression (0.865) is close to the best r2_score of ridge regression (0.890), which suggests what the ridge regression learned most from the dataset is the feature information about deviation from normal distribution")

print(" - For prediction accuracy, the best model is ridge model using data
with one hot encoding
	- For model explainability, the best model is linear model using data
with one hot encoding and skew transformation")



