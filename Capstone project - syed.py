#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("D:\Imarticus Learning\ML\Project\online+shoppers+purchasing+intention+dataset(2)\online_shoppers_intention.csv")


# In[3]:


df #Full Data


# ### Description of the data

# In[4]:


df.describe()


# ### Information about the data

# In[5]:


df.info() #info of the data


# ### Dimensions of the Data

# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


print('The basic distribution of the dataset', df.describe())


# In[10]:


df['Administrative']=df['Administrative'].astype(object)
df['Informational']=df['Informational'].astype(object)
df['ProductRelated']=df['ProductRelated'].astype(object)
df['OperatingSystems']=df['OperatingSystems'].astype(object)
df['Browser']=df['Browser'].astype(object)
df['Region']=df['Region'].astype(object)
df['TrafficType']=df['TrafficType'].astype(object)
df['SpecialDay']=df['SpecialDay'].astype(object)
df.info()


# In[11]:


df.info()


# #### Exploratory Data Analysis
# 
# Univariate Analysis

# In[12]:


df['VisitorType'].value_counts()


# In[13]:


# Checking the unique Browsers
df['Browser'].value_counts()


# In[14]:


plt.figure(figsize = (18,7))
sns.countplot(df['Administrative'], color = "red")
plt.show()


# In[15]:


plt.figure(figsize = (18,7))
sns.distplot(df['Administrative_Duration'], color = "crimson")
plt.show()


# In[16]:


plt.figure(figsize = (18,7))
sns.countplot(df['Informational'], color = "crimson")
plt.show()


# In[17]:


plt.figure(figsize = (18,7))
sns.distplot(df['ExitRates'], color = "crimson")
plt.show()


# In[18]:


plt.figure(figsize = (18,7))
sns.distplot(df['ProductRelated_Duration'], color = "crimson")
plt.show()


# In[19]:


plt.figure(figsize = (18,7))
sns.distplot(df['BounceRates'], color = "crimson")
plt.show()


# In[20]:


plt.figure(figsize = (16,7))
sns.countplot(df['Informational_Duration'], color = "crimson")
plt.show()


# In[21]:


plt.figure(figsize = (18,7))
sns.countplot(df['ProductRelated'], color = "crimson")
plt.show()


# In[22]:


sns.pairplot(df,x_vars=['BounceRates','ExitRates'],y_vars=['BounceRates','ExitRates'],hue='Revenue',diag_kind='kde')
plt.show()


# In[23]:


plt.figure(figsize = (18,7))
sns.countplot(df['ProductRelated'], color = "crimson")
plt.show()


# In[24]:


from sklearn.preprocessing import quantile_transform


# In[25]:


import scipy.stats as stats
pro_duratn = quantile_transform(df[['ProductRelated_Duration']], output_distribution='normal',random_state=0, copy='warn').flatten()
inf_duration= quantile_transform(df[['Informational_Duration']], output_distribution='uniform',random_state=0, copy='warn').flatten()
adm_duration= quantile_transform(df[['Administrative_Duration']], output_distribution='normal',random_state=0, copy='warn').flatten()
bounce_duration= quantile_transform(df[['BounceRates']], output_distribution='uniform',random_state=0, copy='warn').flatten()


# In[26]:


plt.figure(figsize = (18,7))
sns.distplot(df['BounceRates'], color = "crimson")
plt.show()


# In[27]:


plt.figure(figsize = (18,7))
sns.distplot(df['ExitRates'], color = "crimson")
plt.show()


# In[28]:


sns.distplot(pro_duratn, color = "crimson")
plt.show()


# In[29]:


sns.distplot(inf_duration, color = "crimson")
plt.show()


# In[30]:


sns.distplot(adm_duration, color = "crimson")
plt.show()


# In[31]:


df['SpecialDay'].value_counts()


# In[32]:


plt.figure(figsize = (18,7))

sns.countplot(df['SpecialDay'], palette = 'pastel')
plt.show()


# In[33]:


plt.figure(figsize = (18,7))
sns.countplot(df['OperatingSystems'], palette = 'pastel')
plt.show()


# In[34]:


plt.figure(figsize = (18,7))
sns.countplot(df['Browser'], palette = 'pastel')
plt.show()


# In[35]:


plt.figure(figsize = (18,7))
sns.countplot(df['Region'], palette = 'pastel')
plt.show()


# In[36]:


plt.figure(figsize = (18,7))
sns.countplot(df['TrafficType'], palette = 'pastel')
plt.show()


# In[37]:


print(df['Month'].value_counts())
print(df['VisitorType'].value_counts())


# In[38]:


# Month
df['Month'].value_counts().plot(kind = "bar")

#By plt
plt.xticks(rotation = 90)
plt.show()


# In[39]:


# By Sns
sns.countplot(x = "Month", data = df)


# In[40]:


# VisitoType
df['VisitorType'].value_counts().plot(kind = "bar")

#by plt
plt.xticks(rotation = 90)
plt.show()


# In[41]:


# By Sns
sns.countplot(x = "VisitorType", data = df)


# ### Checking the Distribution of customers on Revenue
# 
# 

# In[42]:


plt.figure(figsize = (10,7))

plt.subplot(1, 2, 1)
sns.countplot(df['Weekend'], palette = 'pastel')
plt.title('Buy or Not', fontsize = 20)
plt.xlabel('Revenue or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)

# checking the Distribution of customers on Weekend
plt.subplot(1, 2, 2)
sns.countplot(df['Weekend'], palette = 'inferno')
plt.title('Purchase on Weekends', fontsize = 20)
plt.xlabel('Weekend or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)

plt.show()


# #### Bi-Variate Analysis
# 
# ##### Informational Duration vs revenue
# 
# 

# In[43]:


#plt.figure(figsize = (25,18))

sns.boxplot(df['Revenue'], df['Informational_Duration'], palette = 'rainbow')
plt.title('Info. duration vs Revenue', fontsize = 30)
plt.xlabel('Info. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# ##### Administrative Duration vs revenue
# 
# 

# In[44]:


sns.boxplot(df['Revenue'], df['Administrative_Duration'], palette = 'pastel')
plt.title('Admn. duration vs Revenue', fontsize = 25)
plt.xlabel('Admn. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# #### Product related duration vs revenue
# 
# 

# In[45]:


sns.boxplot(df['Revenue'], df['ProductRelated_Duration'], palette = 'dark')
plt.title('Product Related duration vs Revenue', fontsize = 20)
plt.xlabel('Product Related duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# #### Exit rate vs Revenue

# In[46]:


sns.boxplot(df['Revenue'], df['ExitRates'], palette = 'rainbow')
plt.title('ExitRates vs Revenue', fontsize = 25)
plt.xlabel('ExitRates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# #### Page values vs Revenue

# In[47]:


sns.stripplot(df['Revenue'], df['PageValues'], palette = 'autumn')
plt.title('PageValues vs Revenue', fontsize = 25)
plt.xlabel('PageValues', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# ##### Bounce rates vs Revenue

# In[48]:


sns.stripplot(df['Revenue'], df['BounceRates'], palette = 'magma')
plt.title('Bounce Rates vs Revenue', fontsize = 25)
plt.xlabel('Boune Rates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.figure(figsize=(15,9))
plt.show()


# In[49]:


data = pd.crosstab(df['VisitorType'], df['Revenue'])
data.plot(kind = 'bar', stacked = True, figsize = (6, 3), color = ['lightgreen', 'green'])
plt.title('Visitor Type vs Revenue')
plt.show()


# ### Chi-Square
# To determine whether there is a statistically significance of Various Fetaures we use Chi-Square Test
# 
# Weekend

# In[50]:


df_w = df[['Weekend','Revenue']]
df_w.head()


# In[51]:


df_w1 = pd.get_dummies(df_w)


# In[52]:


df_w1


# In[53]:


df_w1.head()


# In[54]:


df_w1.Weekend = df_w1.Weekend.map({False : 0, True : 1})


# In[55]:


df_w1.Revenue = df_w1.Revenue.map({False : 0, True :1})


# In[56]:


df_w1.head()


# In[57]:


from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[58]:


ct=pd.crosstab(df_w.Weekend, df_w.Revenue)
ct


# In[59]:


nn=np.array(ct)
nn


# ### Hypothesis
# 
# HO: The variables are independent.
# 
# HA: The variables are dependent.

# In[60]:


stat, p, dof, expected = chi2_contingency(nn)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print(' Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail)')


# In[61]:


df.columns


# ### Visitor Type
# 

# In[62]:


df_vt = pd.crosstab(df.VisitorType, df.Revenue)
df_vt


# In[63]:


vt = np.array(df_vt)
vt


# In[64]:


stat, p, dof, expected = chi2_contingency(vt)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# Traffic Type
# 

# In[65]:


df_tt = pd.crosstab(df.TrafficType, df.Revenue)
df_tt.plot(kind = 'bar')


# In[66]:


df_tt.TrafficType = df.TrafficType.replace(to_replace = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], value = 5)


# In[67]:


df_tt = pd.crosstab(df_tt.TrafficType, df.Revenue)
df_tt.plot(kind = 'bar')


# In[68]:


df.TrafficType.nunique()


# In[69]:


tt = np.array(df_tt)
tt


# In[70]:


stat, p, dof, expected = chi2_contingency(tt)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# ##### Region

# In[71]:


df_r = pd.crosstab(df.Region, df.Revenue)
df_r.plot(kind = 'bar')
plt.show()


# In[72]:


df_r


# In[73]:


df_r.iloc[5,:]


# In[74]:


df_r.Region = df.Region.replace(to_replace = [5,6,7,8,9], value = 5)


# In[75]:


print(df.Region.nunique())
print(df_r.Region.unique())


# In[76]:


df_r.Region.unique()


# In[77]:


df_r = pd.crosstab(df_r.Region, df.Revenue)
df_r.plot(kind = 'bar')
plt.show()


# In[78]:


rg = np.array(df_r)
rg


# In[79]:


stat, p, dof, expected = chi2_contingency(rg)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# #### Browser

# In[80]:


df_b = pd.crosstab(df.Browser, df.Revenue)
df_b.plot(kind = 'bar')
plt.show()


# In[81]:


df_b.Browser = df.Browser.replace(to_replace = [3,4,5,6,7,8,9,10,11,12,13], value = 3)


# In[82]:


print(df.Browser.nunique())
print(df_b.Browser.unique())


# In[83]:


df_b = pd.crosstab(df_b.Browser, df.Revenue)
df_b.plot(kind = 'bar')
plt.show()


# In[84]:


b = np.array(df_b)
b


# In[85]:


stat, p, dof, expected = chi2_contingency(b)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# ##### Operating System

# In[86]:


df_osy = pd.crosstab(df.OperatingSystems, df.Revenue)
df_osy.plot(kind = 'bar')
plt.show()


# In[87]:


df_osy


# In[88]:


df_osy.OperatingSystems = df.OperatingSystems.replace(to_replace = [4,5,6,7,8], value =4)


# In[89]:


df_osy = pd.crosstab(df_osy.OperatingSystems, df.Revenue)
df_osy


# In[90]:


os = np.array(df_osy)
os


# In[91]:


stat, p, dof, expected = chi2_contingency(os)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# ####  Month
# 

# In[92]:


df_m = pd.crosstab(df.Month, df.Revenue)
df_m.plot(kind = 'bar')
plt.show()


# In[93]:


df_m.Month = df.Month.replace(to_replace = ['Aug','Feb',' Jul','June','Oct', 'Sep'], value = 'Rest')


# In[94]:


df_m = pd.crosstab(df_m.Month, df.Revenue)
df_m.plot(kind = 'bar')
plt.show()


# In[95]:


m = np.array(df_m)
m


# In[96]:


stat, p, dof, expected = chi2_contingency(m)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# #### Special Day

# In[97]:


df_sdy = pd.crosstab(df.SpecialDay, df.Revenue)
df_sdy.plot(kind = 'bar')
plt.show()


# In[98]:


df_sdy.SpecialDay = df.SpecialDay.replace(to_replace = [0.2,0.4,0.6,0.8,1.0], value = 1.0)


# In[99]:


df_sdy = pd.crosstab(df_sdy.SpecialDay, df.Revenue)
df_sdy.plot(kind = 'bar')
plt.show()


# In[100]:


sd = np.array(df_sdy)
sd


# In[101]:


stat, p, dof, expected = chi2_contingency(sd)
print('dof=%d' % dof)
print(expected)
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')
# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


# In[102]:


sns.pairplot(df,x_vars=['BounceRates','ExitRates'],y_vars=['BounceRates','ExitRates'],hue='Revenue',diag_kind='kde')
plt.show()


# In[103]:


df.isnull().sum()[df.isnull().sum()>0]
df.isnull().sum()


# ###  Outlier Treatment
# 
# 

# In[104]:


# For Administrative_Duaration
q1_adm=np.quantile(df.Administrative_Duration,0.25)
q3_adm=np.quantile(df.Administrative_Duration,0.75)
iqr_adm=q3_adm-q1_adm
ll=q1_adm-(1.5*iqr_adm)
ul=q3_adm+(1.5*iqr_adm)
df_ad_out=df[(df.Administrative_Duration<ll) | (df.Administrative_Duration>ul)]
df_ad_out.shape


# In[105]:


# for Informational_Duration
q1_inf=np.quantile(df.Informational_Duration,0.25)
q3_inf=np.quantile(df.Informational_Duration,0.75)
iqr_inf=q3_inf-q1_inf
ll=q1_inf-(1.5*iqr_inf)
ul=q3_inf+(1.5*iqr_inf)
df_inf_out=df[(df.Informational_Duration<ll) | (df.Informational_Duration>ul)]
df_inf_out.shape


# In[106]:


# for Product Related Duaration
q1_pro=np.quantile(df.ProductRelated_Duration,0.25)
q3_pro=np.quantile(df.ProductRelated_Duration,0.75)
iqr_pro=q3_pro-q1_pro
ll=q1_pro-(1.5*iqr_pro)
ul=q3_pro+(1.5*iqr_pro)
df_pro_out=df[(df.ProductRelated_Duration<ll) | (df.ProductRelated_Duration>ul)]
df_pro_out.shape


# In[107]:


# For Bounce Rate
q1_bou=np.quantile(df.BounceRates,0.25)
q3_bou=np.quantile(df.BounceRates,0.75)
iqr_bou=q3_bou-q1_bou
ll=q1_bou-(1.5*iqr_bou)
ul=q3_bou+(1.5*iqr_bou)
df_bou_out=df[(df.BounceRates<ll) | (df.BounceRates>ul)]
df_bou_out.shape


# In[108]:


# for Exit Rate
q1_ex=np.quantile(df.ExitRates,0.25)
q3_ex=np.quantile(df.ExitRates,0.75)
iqr_ex=q3_ex-q1_ex
ll=q1_ex-(1.5*iqr_ex)
ul=q3_ex+(1.5*iqr_ex)
df_ex_out=df[(df.ExitRates<ll) | (df.ExitRates>ul)]
df_ex_out.shape


# In[109]:


# for Page Values
q1_pg=np.quantile(df.PageValues,0.25)
q3_pg=np.quantile(df.PageValues,0.75)
iqr_pg=q3_pg-q1_pg
ll=q1_pg-(1.5*iqr_pg)
ul=q3_pg+(1.5*iqr_pg)
df_pg_out=df[(df.PageValues<ll) | (df.PageValues>ul)]
df_pg_out.shape


# In[110]:


dff=pd.DataFrame()


# In[111]:


dff['Administrative_Duration']=df.index.isin(df_ad_out.index)
dff['Informational_Duration']=df.index.isin(df_inf_out.index)
dff['ProductRelated_Duration']=df.index.isin(df_pro_out.index)
dff['BounceRates']=df.index.isin(df_bou_out.index)
dff['ExitRates']=df.index.isin(df_ex_out.index)
dff['PageValues']=df.index.isin(df_pg_out.index)


# In[112]:


plt.figure(figsize=(20,15))
sns.heatmap(dff , yticklabels = False , cbar = False , cmap = 'viridis')
plt.show()


# ##### _Above Heatmap is For Variables Administrative_Duration, Informational_Duration, ProductRelatedDuration, BounceRates, ExitRates, PageValues. Inwhich Yellow lines Represents the presence of Outliers...

# In[113]:


# Converting Booleans into 1's and 0's
bool_map={True:1,False:0}
df.Weekend.replace(bool_map,inplace=True)
df.Revenue.replace(bool_map,inplace=True)


# In[114]:


df.head()


# In[115]:


dff.head()


# In[116]:


dff['multi'] = ['Y' if x >= 4 else 'N' for x in np.sum(dff.values == True, 1)]


# In[117]:


dff['multi'] = ['Y' if x >= 4 else 'N' for x in np.sum(dff.values == True, 1)]


# In[118]:


df_new=df[dff['multi']=='N']
df_new.shape


# In[119]:


# Converting Booleans into 1's and 0's
bool_map={True:1,False:0}
df_new.Weekend.replace(bool_map,inplace=True)
df_new.Revenue.replace(bool_map,inplace=True)
df_new.head()


# In[120]:


# Replacing the Outliers with NAN
df_new.loc[(dff['Administrative_Duration']==True),'Administrative_Duration']=np.NAN
df_new.loc[(dff['Informational_Duration']==True),'Informational_Duration']=np.NAN
df_new.loc[(dff['ProductRelated_Duration']==True),'ProductRelated_Duration']=np.NAN
df_new.loc[(dff['BounceRates']==True),'BounceRates']=np.NAN
df_new.loc[(dff['PageValues']==True),'PageValues']=np.NAN
#df_new=df_new.drop('ExitRates',axis=1)


# In[121]:


df_new.isnull().sum()[df_new.isnull().sum()>0]


# In[122]:


imp_col=df_new.isnull().sum()[df_new.isnull().sum()>0].index


# In[123]:


# Creating dummy Variables
df_dum=pd.get_dummies(df_new)
df_dum.head()


# #### After Converting the Outliers into NAN, we need to assign values to this null/missing Values. Here we are using MICE(Missing Imputaions through Chained Equations) technique for imputing the missing values.

# In[162]:


get_ipython().system('pip install impyute')
get_ipython().system('pip install -U numpy')


# In[128]:


from sklearn.experimental import enable_iterative_imputer


# In[129]:


from sklearn.impute import IterativeImputer


# In[131]:


imputed_values = IterativeImputer(verbose=1).fit_transform(df_dum)


# In[137]:


imputed_values


# In[124]:


from impyute.imputation.cs import mice


# In[133]:


df_dum.values.astype('float')


# In[134]:


imputed_df=mice(df_dum.values.astype(np.float))


# In[138]:


imputed_values=pd.DataFrame(imputed_values,columns=df_dum.columns)


# In[139]:


imputed_values.head()


# In[140]:


X=imputed_values.drop(['Revenue','ExitRates'],axis=1)
Y=imputed_values.Revenue
#Y.value_counts(normalize=True)


# In[141]:


from sklearn.metrics import f1_score,cohen_kappa_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# #### Spliting the data into train and split

# In[142]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)


# ##### Our Target Variable is Imbalanced First we train model without bakacing the target vairble and
# 
# Later we will train the model using Oversampling technique SMOTE (Synthetic Minority Over-sampling Technique) From imblearn library
# 
# ###### FYI Target is revenue

# In[145]:


#Without SMOTE
imputed_values['Revenue'].value_counts().plot(kind='bar')
plt.show()


# ### Logistics Regression

# In[146]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
log=LogisticRegression(C=0.005994,penalty='l1',solver='liblinear')
log.fit(x_train,y_train)
print('Train score:',log.score(x_train,y_train))
print('Test score:',log.score(x_test,y_test))
#log.C_


# In[147]:


log_pred=log.predict(x_test)
print('F1 Score:',f1_score(y_test,log_pred))
print('Kappa Score:',cohen_kappa_score(y_test,log_pred))
print('Classification report:\n',classification_report(y_test,log_pred))


# In[148]:


from sklearn.metrics import roc_curve,auc
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# In[154]:


fpr,tpr,thresh=roc_curve(y_test,log_pred)
auc_log=auc(fpr,tpr)


#this code calculates the ROC curve and AUC (Area Under the Curve) for a decision tree classifier's 
#predictions on the test data. ROC curve and AUC are commonly used metrics to evaluate the performance of binary classification 
#models. AUC provides a single scalar value indicating the overall performance of the model: the higher the AUC, 
#the better the model's ability to distinguish between positive and negative classes.


# ### Decision Tree

# In[150]:


dt=DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
print('Train score:',dt.score(x_train,y_train))
print('Test score:',dt.score(x_test,y_test))


# In[151]:


dt_pred=dt.predict(x_test)
print('F1 Score:',f1_score(y_test,dt_pred))
print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))
print('Classification report:\n',classification_report(y_test,dt_pred))


# In[155]:


fpr_dt,tpr_dt,thresh=roc_curve(y_test,dt_pred)
auc_dt=auc(fpr,tpr)


# ### Random Forest

# In[156]:


rf_sm=RandomForestClassifier(max_depth=6)
rf_sm.fit(x_train,y_train)
print('Train score:',rf_sm.score(x_train,y_train))
print('Test score:',rf_sm.score(x_test,y_test))


# In[157]:


rf_sm_pred=rf_sm.predict(x_test)
print('F1 Score:',f1_score(y_test,rf_sm_pred))
print('Kappa Score:',cohen_kappa_score(y_test,rf_sm_pred))
print('Classification report:\n',classification_report(y_test,rf_sm_pred))


# In[158]:


fpr_rf,tpr_rf,thresh=roc_curve(y_test,rf_sm_pred)
auc_rf=auc(fpr,tpr)


# In[159]:


plt.plot(fpr,tpr, label='LR(area = %0.2f)' % auc_log,color='red')
plt.plot(fpr_dt, tpr_dt, label='DT(area = %0.2f)' % auc_dt,color='green')
plt.plot(fpr_rf, tpr_rf, label='RF(area = %0.2f)' % auc_rf,color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ### Naive bayes

# In[160]:


gnb=GaussianNB()
gnb.fit(x_train,y_train)
print('Train score:',gnb.score(x_train,y_train))
print('Test score:',gnb.score(x_test,y_test))


# In[161]:


nb_pred=gnb.predict(x_test)
fpr2,tpr2,thresh=roc_curve(y_test,nb_pred)
auc2=auc(fpr2,tpr2)


# In[162]:


dt=DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
print('Train score:',dt.score(x_train,y_train))
print('Test score:',dt.score(x_test,y_test))


# In[163]:


dt_pred=dt.predict(x_test)
print('F1 Score:',f1_score(y_test,dt_pred))
print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))


# In[164]:


fpr3,tpr3,thresh=roc_curve(y_test,dt_pred)
auc3=auc(fpr3,tpr3)


# In[166]:


rf_sm=RandomForestClassifier(max_depth=6)
rf_sm.fit(x_train,y_train)
print('Train score:',rf_sm.score(x_train,y_train))
print('Test score:',rf_sm.score(x_test,y_test))


# In[167]:


rf_pred_sm=rf_sm.predict(x_test)
print('F1 Score:',f1_score(y_test,rf_pred_sm))
print('Kappa Score:',cohen_kappa_score(y_test,rf_pred_sm))


# In[168]:


fpr4,tpr4,thresh=roc_curve(y_test,rf_pred_sm)
auc4=auc(fpr4,tpr4)


# In[ ]:





# #### With SMOTE

# In[175]:


get_ipython().system('pip install -U imbalanced-learn')


# In[171]:


conda install -c conda-forge imbalanced-learn


# In[173]:


from imblearn.over_sampling import SMOTE


# In[177]:


from imblearn.over_sampling import RandomOverSampler
smote=RandomOverSampler(random_state=42)
X_new,Y_new=smote.fit_resample(X,Y)
X_new=pd.DataFrame(X_new,columns=X.columns)
Y_new=pd.DataFrame(Y_new,columns=['Revenue'])
X_new.head()


# In[178]:


Y_new['Revenue'].value_counts().plot(kind='bar')
plt.show()


# In[179]:


x_train,x_test,y_train,y_test=train_test_split(X_new,Y_new,test_size=0.3,random_state=1)


# ### Logistics Regression

# In[180]:


log=LogisticRegression(penalty='l1',solver='liblinear')
log.fit(x_train,y_train)
print('Train score:',log.score(x_train,y_train))
print('Test score:',log.score(x_test,y_test))


# In[181]:


log_pred=log.predict(x_test)
print('F1 Score:',f1_score(y_test,log_pred))
print('Kappa Score:',cohen_kappa_score(y_test,log_pred))
print('Classification report:\n',classification_report(y_test,log_pred))


# In[182]:


fpr1,tpr1,thresh=roc_curve(y_test,log_pred)
auc1=auc(fpr1,tpr1)


# ### Naive bayes

# In[184]:


gnb=GaussianNB()
gnb.fit(x_train,y_train)
print('Train score:',gnb.score(x_train,y_train))
print('Test score:',gnb.score(x_test,y_test))


# In[185]:


nb_pred=gnb.predict(x_test)
fpr2,tpr2,thresh=roc_curve(y_test,nb_pred)
auc2=auc(fpr2,tpr2)


# ### Decision Tree

# In[186]:


dt=DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
print('Train score:',dt.score(x_train,y_train))
print('Test score:',dt.score(x_test,y_test))


# In[190]:


dt_pred=dt.predict(x_test)
print('F1 Score:',f1_score(y_test,dt_pred))
print('Kappa Score:',cohen_kappa_score(y_test,dt_pred))


# In[191]:


fpr3,tpr3,thresh=roc_curve(y_test,dt_pred)
auc3=auc(fpr3,tpr3)


# ### Random Forest

# In[192]:


rf_sm=RandomForestClassifier(max_depth=6)
rf_sm.fit(x_train,y_train)
print('Train score:',rf_sm.score(x_train,y_train))
print('Test score:',rf_sm.score(x_test,y_test))


# In[193]:


rf_pred_sm=rf_sm.predict(x_test)
print('F1 Score:',f1_score(y_test,rf_pred_sm))
print('Kappa Score:',cohen_kappa_score(y_test,rf_pred_sm))


# In[194]:


fpr4,tpr4,thresh=roc_curve(y_test,rf_pred_sm)
auc4=auc(fpr4,tpr4)


# In[195]:


imp=pd.DataFrame(rf_sm.feature_importances_, columns = ["Imp"], index =x_train.columns)


# In[196]:


plt.figure(figsize=(20,8))
imp.sort_values('Imp',ascending=False).head(70).plot(kind='bar')
plt.xticks(rotation=80)
plt.show()


# In[197]:


imp.sort_values('Imp',ascending=False).head()


# In[198]:


imp.sort_values('Imp',ascending=False).head()


# In[200]:


len(imp['Imp'])


# In[201]:


imp2=imp[imp["Imp"]>0.0005]
len(imp2['Imp'])


# In[202]:


imp2.sort_values('Imp',ascending=False).index


# In[203]:


xnew=X_new[imp2.index]
x_train,x_test,y_train,y_test=train_test_split(xnew,Y_new,test_size=0.3,random_state=1)


# ### Logistics Regression

# In[205]:


log=LogisticRegression(penalty='l1',solver='liblinear')
log.fit(x_train,y_train)
print('Train score:',log.score(x_train,y_train))
print('Test score:',log.score(x_test,y_test))


# In[206]:


log_sm1_pred=log.predict(x_test)
print('F1 Score:',f1_score(y_test,log_sm1_pred))
print('Kappa Score:',cohen_kappa_score(y_test,log_sm1_pred))


# ### Desicion Tree

# In[207]:


dt=DecisionTreeClassifier(max_depth=6)
dt.fit(x_train,y_train)
print('Train score:',dt.score(x_train,y_train))
print('Test score:',dt.score(x_test,y_test))


# In[208]:


dt_sm1_pred_sm=dt.predict(x_test)
print('F1 Score:',f1_score(y_test,dt_sm1_pred_sm))
print('Kappa Score:',cohen_kappa_score(y_test,dt_sm1_pred_sm))


# ### Random Forest

# In[210]:


rf_sm1=RandomForestClassifier(n_estimators=50,max_depth=16)
rf_sm1.fit(x_train,y_train)
print('Train score:',rf_sm1.score(x_train,y_train))
print('Test score:',rf_sm1.score(x_test,y_test))


# In[211]:


rf_sm1_pred_sm=rf_sm1.predict(x_test)
print('F1 Score:',f1_score(y_test,rf_sm1_pred_sm))
print('Kappa Score:',cohen_kappa_score(y_test,rf_sm1_pred_sm))


# In[212]:


fpr5,tpr5,thresh=roc_curve(y_test,rf_sm1_pred_sm)
auc5=auc(fpr5,tpr5)


# ### Gradient boosting

# In[213]:


gb=GradientBoostingClassifier(n_estimators=50,max_depth=5)
gb.fit(x_train,y_train)
print('Train score:',gb.score(x_train,y_train))
print('Test score:',gb.score(x_test,y_test))


# In[214]:


gb_pred_sm=gb.predict(x_test)
print('F1 Score:',f1_score(y_test,gb_pred_sm))
print('Kappa Score:',cohen_kappa_score(y_test,gb_pred_sm))


# In[215]:


fpr6,tpr6,thresh=roc_curve(y_test,gb_pred_sm)
auc6=auc(fpr6,tpr6)


# ### Plotting ROC Curve for Different Models¶

# In[216]:


plt.plot(fpr1,tpr1, label='LR(area = %0.2f)' % auc1,color='red')
plt.plot(fpr2, tpr2, label='NB(area = %0.2f)' % auc2,color='black')
plt.plot(fpr3, tpr3, label='DT(area = %0.2f)' % auc3,color='magenta')
plt.plot(fpr4, tpr4, label='RF(area = %0.2f)' % auc4,color='blue')
plt.plot(fpr6, tpr6, label='GB(area = %0.2f)' % auc6,color='pink')
plt.plot(fpr6, tpr6, label='RF with FS(area = %0.2f)' % auc5,color='green')


plt.plot([0, 1], [0, 1], 'k--',color='grey')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ##### As observed from findings it is clear that the performance improved with SMOTE. Among all models Random Forest With Feature Selection gives best accuracy metrics.
# 
# As seen from accuracy metrics and ROC, we can see that Random Forest is best among all as it gives 98% area under, F1 score as 0.90
# 
# Transformation improved accuracies for Logistic and Naïve Bayes but it hardly improved Random Forest and Decision Tree. So, we go ahead without transformations as it hardly improves any accuracy.
# 
# Random Forest was used for feature selection and the important features given by Random Forest are supported by the EDA analysis done.

# In[217]:


y = df['Revenue']


# In[227]:


from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# #### Cross Validating Different models

# In[228]:


models = []
models.append(('LR', LogisticRegression(penalty='l1',solver='liblinear')))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, xnew, Y_new, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (std=%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# #### CONCLUSION
# SMOTE has improved our accuracy as the dataset was unbalanced dataset.
# 
# Dataset with outliers (without treating outliers) gave us better results.
# 
# Transformation did not improve model performance significantly
# 
# So, our final model is the Random Forest employed on top of SMOTE, with outliers and without transformations.

# In[ ]:




