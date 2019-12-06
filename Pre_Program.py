#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data = pd.read_csv('auto-mpg.csv',index_col='car name')


# In[15]:


data.head()
print(data.index)
print(data.columns)


# In[16]:


data.shape


# In[17]:


data.isnull().any()


# In[18]:


data.dtypes


# In[19]:


data.horsepower.unique()


# In[20]:


data = data[data.horsepower != '?']


# In[21]:


print('?' in data.horsepower)


# In[22]:


data.shape


# In[24]:


data.dtypes


# In[25]:


data.horsepower = data.horsepower.astype('float')
data.dtypes


# In[26]:


data.describe()


# In[27]:


data.mpg.describe()


# In[28]:


sns.distplot(data['mpg'])


# In[29]:


print("Skewness: %f" % data['mpg'].skew())
print("Kurtosis: %f" % data['mpg'].kurt())


# In[30]:


def scale(a):
    b = (a-a.min())/(a.max()-a.min())
    return b


# In[31]:


data_scale = data.copy()


# In[32]:


data_scale ['displacement'] = scale(data_scale['displacement'])
data_scale['horsepower'] = scale(data_scale['horsepower'])
data_scale ['acceleration'] = scale(data_scale['acceleration'])
data_scale ['weight'] = scale(data_scale['weight'])
data_scale['mpg'] = scale(data_scale['mpg'])


# In[33]:


data_scale.head()


# In[34]:


data['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])
data_scale['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])


# In[35]:


data_scale.head()


# In[36]:


var = 'Country_code'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# In[37]:


var = 'model year'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# In[38]:


var = 'cylinders'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# In[39]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# In[40]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','mpg']
corrmat = data[factors].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# In[42]:


#scatterplot
sns.set()
sns.pairplot(data, size = 2.0,hue ='Country_code')
plt.show()


# In[43]:


data.index


# In[44]:


data[data.index.str.contains('subaru')].index.str.replace('(.*)', 'subaru dl')


# In[45]:


data['Company_Name'] = data.index.str.extract('(^.*?)\s')


# In[46]:


data['Company_Name'] = data['Company_Name'].replace(['volkswagen','vokswagen','vw'],'VW')
data['Company_Name'] = data['Company_Name'].replace('maxda','mazda')
data['Company_Name'] = data['Company_Name'].replace('toyouta','toyota')
data['Company_Name'] = data['Company_Name'].replace('mercedes','mercedes-benz')
data['Company_Name'] = data['Company_Name'].replace('nissan','datsun')
data['Company_Name'] = data['Company_Name'].replace('capri','ford')
data['Company_Name'] = data['Company_Name'].replace(['chevroelt','chevy'],'chevrolet')
data['Company_Name'].fillna(value = 'subaru',inplace=True)  ## Strin methords will not work on null values so we use fillna()


# In[48]:


var = 'Company_Name'
data_plt = pd.concat([data_scale['mpg'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(20,10))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)


# In[49]:


data.Company_Name.isnull().any()


# In[50]:


var='mpg'
data[data[var]== data[var].min()]


# In[51]:


data[data[var]== data[var].max()]


# In[52]:


var='displacement'
data[data[var]== data[var].min()]


# In[53]:


data[data[var]== data[var].max()]


# In[54]:


var='horsepower'
data[data[var]== data[var].min()]


# In[55]:


data[data[var]== data[var].max()]


# In[56]:


var='weight'
data[data[var]== data[var].min()]


# In[57]:


data[data[var]== data[var].max()]


# In[58]:


var='acceleration'
data[data[var]== data[var].min()]


# In[59]:


data[data[var]== data[var].max()]


# In[60]:


var = 'horsepower'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[61]:


var = 'displacement'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[62]:


var = 'weight'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[63]:


var = 'acceleration'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))


# In[64]:


data['Power_to_weight'] = ((data.horsepower*0.7457)/data.weight)


# In[65]:


data.sort_values(by='Power_to_weight',ascending=False ).head()


# In[66]:


data.head()


# In[67]:


from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# In[68]:


factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
X = pd.DataFrame(data[factors].copy())
y = data['mpg'].copy()


# In[69]:


X = StandardScaler().fit_transform(X)


# In[70]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)
X_train.shape[0] == y_train.shape[0]


# In[71]:


regressor = LinearRegression()


# In[72]:


regressor.get_params()


# In[73]:


regressor.fit(X_train,y_train)


# In[74]:


y_predicted = regressor.predict(X_test)


# In[75]:


rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
rmse


# In[76]:


gb_regressor = GradientBoostingRegressor(n_estimators=4000)
gb_regressor.fit(X_train,y_train)


# In[77]:


gb_regressor.get_params()


# In[78]:


y_predicted_gbr = gb_regressor.predict(X_test)


# In[79]:


rmse_bgr = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr))
rmse_bgr


# In[80]:


fi= pd.Series(gb_regressor.feature_importances_,index=factors)
fi.plot.barh()


# In[81]:


from sklearn.decomposition import PCA


# In[82]:


pca = PCA(n_components=2)


# In[83]:


pca.fit(data[factors])


# In[84]:


pca.explained_variance_ratio_


# In[85]:


pca1 = pca.components_[0]
pca2 = pca.components_[1]


# In[86]:


transformed_data = pca.transform(data[factors])


# In[88]:


pc1 = transformed_data[:,0]
pc2 = transformed_data[:,1]


# In[89]:


plt.scatter(pc1,pc2)


# In[90]:


c = pca.inverse_transform(transformed_data[(transformed_data[:,0]>0 )& (transformed_data[:,1]>250)])


# In[91]:


factors


# In[92]:


c


# In[93]:


data[(data['model year'] == 70 )&( data.displacement>400)]


# In[94]:


cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)
params = {'n_estimators' : list(range(40,61)),
         'max_depth' : list(range(1,10)),
         'learning_rate' : [0.1,0.2,0.3] }
grid = GridSearchCV(gb_regressor, params,cv=cv_sets,n_jobs=4)


# In[95]:


grid = grid.fit(X_train, y_train)


# In[96]:


grid.best_estimator_


# In[97]:


gb_regressor_t = grid.best_estimator_


# In[98]:


gb_regressor_t.fit(X_train,y_train)


# In[99]:


y_predicted_gbr_t = gb_regressor_t.predict(X_test)


# In[100]:


rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))
rmse


# In[101]:


data.duplicated().any()


# In[ ]:




