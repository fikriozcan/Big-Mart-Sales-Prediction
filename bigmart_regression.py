
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
from pandas import set_option
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display_html
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['patch.facecolor'] = 'b'

# rainbow colors
rb = []
colors = plt.cm.rainbow(np.linspace(0,1,18))
for c in colors:
    rb.append(c)
rb = reversed(rb)
rb = list(rb)

# viridis colors
vd = []
colors = plt.cm.GnBu(np.linspace(0,1,6))
for c in colors:
    vd.append(c)
vd = list(vd)


# In[9]:


file ='train.xlsx'
sales = pd.read_excel(file)
sales.info()
sales.head(10)


# In[12]:


sales.isnull().sum()


# In[17]:





# In[24]:


fig, ax = plt.subplots(3,2, figsize=(20,20))
sns.barplot(x='Item_Fat_Content', y='index', data = sales['Item_Fat_Content'].value_counts().reset_index(), ax=ax[0,0])
ax[0,0].set_title('Number of instances in Fat Content')
ax[0,0].set_ylabel('Item Fat Content ')
ax[0,0].set_xlabel('Number of variables in Fat Content')

sns.barplot(x='Item_Type', y='index', data = sales['Item_Type'].value_counts().reset_index(), ax=ax[0,1])
ax[0,1].set_title('Total Amount of each Item')
ax[0,1].set_ylabel('Item Type')
ax[0,1].set_xlabel('Total Amount of each Item')


sns.barplot(x='Outlet_Type', y='index', data = sales['Outlet_Type'].value_counts().reset_index(), ax=ax[1,0])
ax[1,0].set_title('Number of Outlet by Type ')
ax[1,0].set_ylabel('Outlet Type')
ax[1,0].set_xlabel('Number of Outlet by Type')


sns.barplot(x='Outlet_Size', y='index', data = sales['Outlet_Size'].value_counts().reset_index(), ax=ax[1,1])
ax[1,1].set_title('Number of Outlet by Size')
ax[1,1].set_ylabel('Outlet Size')
ax[1,1].set_xlabel('Number of Outlet by Size')


sns.distplot( sales["Item_Visibility"] , color="black", label="Item_Visibility",ax=ax[2,0])


sns.barplot(x='Outlet_Location_Type', y='index', data = sales['Outlet_Location_Type'].value_counts().reset_index(), ax=ax[2,1])
ax[2,1].set_title('Number of Outlet Location Type')
ax[2,1].set_ylabel('Outlet Location Type')
ax[2,1].set_xlabel('Number of Outlet Location Type')


plt.tight_layout()
plt.show()


# In[25]:


sns.distplot( sales["Item_Outlet_Sales"] , color="red", label="Sales")


# In[26]:


sales.plot.scatter(x='Item_Visibility', y='Item_Outlet_Sales',
                     figsize=(12, 6),
                     title='Product by Sales and Visibility')


# In[27]:


sales.plot.scatter(x='Item_Weight', y='Item_Visibility',
                     figsize=(12, 6),
                     title='Product by Weight and Visibility')


# In[28]:


sales.plot.scatter(x='Item_Outlet_Sales', y='Item_Weight',
                     figsize=(12, 6),
                     title='Product by Weight and Sales')


# In[29]:


## creating new columns called percentage of sales ofr better visuals
sales['percentage_of_sales']= sales['Item_Outlet_Sales'] / sales['Item_Outlet_Sales'].sum()

### creating new columns called year of operation 

sales['Outlet_Establishment_Year'].value_counts
sales['year_of_operation'] = sales['Outlet_Establishment_Year'] - 1985

sales['year_of_operation'].value_counts().plot.bar()


# In[47]:


# Which products sold more? 
item_sales =  sales.groupby('Item_Type')['percentage_of_sales'].sum() # Preparing subset of data for chart
a= item_sales.index.map(lambda x: str(x))
b = item_sales.values

fig,ax = plt.subplots(figsize = (60,30))
sns.barplot(a,b, order=a, palette='viridis',ax=ax)



# In[48]:


## price , sales, item type relationship 
fig, ax = plt.subplots(3,2, figsize=(18,12))

d = {'Item_Outlet_Sales':'Sum', 'Item_MRP':'Mean'}
df=sales.groupby('Item_Type').agg({'Item_Outlet_Sales':'sum', 'Item_MRP':'median'})
print (df)

labels_max = df.sort_values(by='Item_MRP', ascending=False).head(7) # find label for top 7 types for attack
labels_min = df.sort_values(by='Item_MRP', ascending=True).head(3) # find label for last 3 types for attack
label_high = labels_max.index.tolist()
label_low = labels_min.index.tolist()


ax[1,0].scatter(x=df['Item_MRP'], y=df['Item_Outlet_Sales'],s=200,label=df.index, c=rb, alpha=0.7)
for label, x, y in zip(label_high, labels_max['Item_MRP'], labels_max['Item_Outlet_Sales']):
    ax[1,0].annotate(
        label, xy=(x, y), xytext=(-20, -5), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for label, a, b in zip(label_low, labels_min['Item_MRP'], labels_min['Item_Outlet_Sales']):
    ax[1,0].annotate(
        label, xy=(a, b), xytext=(14, 40), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
ax[1,0].set_title('Price & Sales Relationship per item category')

###########################
f = {'Item_Outlet_Sales':'Sum', 'Item_Visibility':'Mean'}
df_1=sales.groupby('Item_Type').agg({'Item_Outlet_Sales':'sum', 'Item_Visibility':'mean'})
print (df_1)
labels_vis_max = df_1.sort_values(by='Item_Visibility', ascending=False).head(7) # find label for top 7 types for attack
labels_vis_min = df_1.sort_values(by='Item_Visibility', ascending=True).head(3) # find label for last 3 types for attack
label_vis_high = labels_vis_max.index.tolist()
label_vis_low = labels_vis_min.index.tolist()

ax[0,0].scatter(x=df_1['Item_Visibility'], y=df_1['Item_Outlet_Sales'],s=200,label=df_1.index, c=rb, alpha=0.7)
for label, x, y in zip(label_vis_high, labels_vis_max['Item_Visibility'], labels_vis_max['Item_Outlet_Sales']):
    ax[0,0].annotate(
        label, xy=(x, y), xytext=(-20, -5), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
for label, a, b in zip(label_vis_low, labels_vis_min['Item_Visibility'], labels_vis_min['Item_Outlet_Sales']):
    ax[0,0].annotate(
        label, xy=(a, b), xytext=(14, 40), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
ax[0,0].set_title('Visibility & Sales Relationship per item category')

##################3
oper_sales = sales.groupby('year_of_operation')['percentage_of_sales'].sum() # Preparing subset of data for chart
x = oper_sales.index.map(lambda x: str(x))
y = oper_sales.values

sns.barplot(x,y, order=x, palette='viridis',ax=ax[0,1])
ax[0,1].set_xlabel('years of operation ')
ax[0,1].set_ylabel('sum of % of sales')
ax[0,1].set_title('Years of operation & Sales Relationship ')

######################

loctype_sales = sales.groupby('Outlet_Location_Type')['percentage_of_sales'].sum() # Preparing subset of data for chart
a = loctype_sales.index.map(lambda x: str(x))
b = loctype_sales.values

sns.barplot(a,b, order=a, palette='viridis',ax=ax[1,1])
ax[1,1].set_xlabel('Outlet Location Type')
ax[1,1].set_ylabel('sum of % of sales')
ax[1,1].set_title('Location Type & Sales Relationship')

outlet_type_sales = sales.groupby('Outlet_Type')['percentage_of_sales'].sum() # Preparing subset of data for chart
c= outlet_type_sales.index.map(lambda x: str(x))
d = outlet_type_sales.values
sns.barplot(c,d, order=c, palette='viridis',ax=ax[2,1])
ax[2,1].set_xlabel('Outlet Type')
ax[2,1].set_ylabel('sum of % of sales')
ax[2,1].set_title('Type of Outlet & Sales Relationship')


outlet_type_sales = sales.groupby('Outlet_Identifier')['percentage_of_sales'].sum() # Preparing subset of data for chart
e= outlet_type_sales.index.map(lambda x: str(x))
f = outlet_type_sales.values
sns.barplot(e,f, order=e, palette='viridis',ax=ax[2,0])
ax[2,0].set_xlabel('Outlets')
ax[2,0].set_ylabel('sum of % of sales')
ax[2,0].set_title('Performance of Each Outlet')



plt.tight_layout()
plt.show()


# In[49]:


sales.hist(bins=50, figsize=(20,15))
plt.show()


# In[52]:


corr_matrix = sales.corr()
corr_matrix["Item_Outlet_Sales"].sort_values(ascending=False)


# In[53]:


df_corr = sales.corr().round(2)
sns.palplot(sns.color_palette('coolwarm', 12))
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.show()


## MODEL BUILDING 
### imputing numeric values with mean 

sales.isnull().sum()

x = sales['Item_Weight'].mean()
sales['Item_Weight'] = (sales['Item_Weight'].fillna(x).round(2))

sales['Outlet_Size']= (sales['Outlet_Size'].fillna(-1))

sales['Item_Fat_Content'].value_counts()
for val in enumerate(sales.loc[ : , 'Item_Fat_Content']):
     if val[1] == "LF" or val[1]== 'low fat' or val[1] == 'Low Fat' :
         sales.loc[val[0],'Item_Fat_Content'] = 'Low Fat'
     elif val[1] == "reg" or val[1]== 'Regular' :
         sales.loc[val[0],'Item_Fat_Content'] = 'Regular'

sales.loc[:, "Outlet_Size"] = pd.factorize(sales.Outlet_Size)[0]
sales['year_of_operation'] = sales['Outlet_Establishment_Year'] - 1985
del sales['Outlet_Establishment_Year']

from sklearn.model_selection import train_test_split
strat_train_set, strat_test_set = train_test_split(
    sales, test_size=0.2, random_state=42, stratify=sales["Item_Type"])

strat_test_set["Item_Type"].value_counts() / len(strat_test_set)
features = strat_train_set.drop("Item_Outlet_Sales", axis=1)
target = strat_train_set["Item_Outlet_Sales"].copy()

################################################ kitaplardan bakıyoruz ama bos value kalmadı 
features.isnull().sum()
sales_non_num = features.loc[:,['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']]
sales_num = features.drop(['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type'],axis = 1)
 ##### sales_non_num

    
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
num_attribs = list(sales_num)
cat_attribs = list(sales_non_num)
full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), num_attribs),
        ("cat", OrdinalEncoder(), cat_attribs),
    ])
features_prepared = full_pipeline.fit_transform(features)

list_full = num_attribs + cat_attribs

features_prepared=pd.DataFrame(features_prepared,columns =list_full )

### sklearn linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(features_prepared, target)
from sklearn.metrics import mean_squared_error
sales_predictions = lin_reg.predict(features_prepared)
lin_mse = mean_squared_error(target, sales_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.metrics import r2_score
r2_score(target, sales_predictions)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, features_prepared, target,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(lin_rmse_scores)


## looking for interactions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

regression = LinearRegression(normalize=True)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
baseline = np.mean(cross_val_score(regression, features_prepared, target, scoring='r2', cv=crossvalidation,
 n_jobs=1))
interactions = list()
for feature_A in features_prepared:
 for feature_B in features_prepared:
  if feature_A > feature_B:
   features_prepared['interaction'] = features_prepared[feature_A] * features_prepared[feature_B]
   score = np.mean(cross_val_score(regression, features_prepared, target, scoring='r2',
    cv=crossvalidation, n_jobs=1))
   if score > baseline:
    interactions.append((feature_A, feature_B, round(score,3)))
    
print(baseline)
print(sorted(interactions))

from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.Series([variance_inflation_factor(features_prepared.values, i) 
               for i in range(features_prepared.shape[1])], 
              index=features_prepared.columns)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(features_prepared, target)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(features_prepared)

final_mse = mean_squared_error(target, final_predictions)

final_rmse = np.sqrt(final_mse)
print(final_rmse)
from sklearn.metrics import r2_score
r2_score(target, final_predictions)
