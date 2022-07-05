#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries
# 

# In[1]:


#Load the libraries
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
import warnings #To avoid any warnings
warnings.filterwarnings(action="ignore")
import datetime as dt # To work with Time date data set


# # 1) Data Understanding

# ## Importing Data

# In[2]:


df = pd.read_csv('loan.csv')
df.head()


# In[3]:


df.info()


# ## Checking for the columns having more than 80% Null values

# In[4]:


df.columns[100*df.isnull().mean() > 80]


# In[5]:


# As this columns have very high percentage of null values, this can be dropped from the analysis
Null_cols = ['mths_since_last_record', 'next_pymnt_d', 'mths_since_last_major_derog',
       'annual_inc_joint', 'dti_joint', 'verification_status_joint',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
       'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il',
       'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
       'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
       'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
       'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
       'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd',
       'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
       'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
       'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
       'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
       'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
       'total_il_high_credit_limit']
df = df.drop(Null_cols, axis = 1)
df.shape


# ## Deleting Customer behaviour columns

# In[6]:


#The following customer behavior variables are not available at the time of loan application. This are post loan approval variables
#thus they cannot be used as predictors for credit approval This variables are listed below and they can be removed
cust_behav_cols = ['delinq_2yrs','earliest_cr_line','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc',
                  'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
                  'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','last_credit_pull_d','application_type']
df = df.drop(cust_behav_cols, axis = 1)
df.shape


# In[7]:


# checking available columns
df.info()


# ## Deleting single valued data Columns

# In[8]:


# It is observed that, many columns have single value e.g. pymnt_plan column has single value 'n'. 
#Such columns are identified and deleted

single_value_cols = ['pymnt_plan', 'collections_12_mths_ex_med', 'policy_code', 'delinq_amnt', 'acc_now_delinq', 'tax_liens',
                      'chargeoff_within_12_mths']
df = df.drop(single_value_cols, axis = 1)
df.shape


# ## Deleting non-essential Columns

# In[9]:


# There are few columns which does not contribute to analysis, this are identified and deleted

non_essential_cols = ['id','member_id','emp_title','url','desc','title','zip_code','mths_since_last_delinq',
                      'initial_list_status']
df = df.drop(non_essential_cols, axis = 1)
df.shape


# In[10]:


# Columns available for analysis
df.info()


# # 2) Data Cleaning and Imputation

# ### Null value detection and imputation

# In[11]:


# checking for null values present in the dataset
100*df.isnull().mean()


# In[12]:


# Null value treatment for 'emp_length' column

# checking value count for emp_length column
df['emp_length'].value_counts()


# In[13]:


# Replacing null values by mode value in emp_length column
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])

# checking null value count
df['emp_length'].isnull().mean()


# In[14]:


# Null value treatment for 'pub_rec_bankruptcies' columnn

# checking value count for pub_rec_bankruptcies column
df['pub_rec_bankruptcies'].value_counts()


# In[15]:


# Replacing null values by mode value in emp_length column
df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(df['pub_rec_bankruptcies'].mode()[0])

# checking null value count
df['pub_rec_bankruptcies'].isnull().mean()


# In[16]:


# checking for null value count again in dataset
100*df.isnull().mean()


# ### int_rate column Treatment

# In[17]:


# checking value count
df['int_rate'].value_counts()


# In[18]:


# removing percentage symbol and converting it into float datatype
df['int_rate'] = df['int_rate'].apply(lambda x: float(x[:-1]))

df['int_rate'].value_counts()


# In[19]:


# Binning int_rate into categories
df['bin_int_rate'] = pd.cut(df['int_rate'], bins=4,precision =0,labels=['5-10','10-15','15-20','20-25'])
df['bin_int_rate'].head()


# ### emp_length column Treatment

# In[20]:


# emp_length dataset value count
df['emp_length'].value_counts()


# In[21]:


# replacing ''< 1 year' employment length by 0 and '10+ years' by 10 and converting data type to 'Integer'
df['emp_length'] = df['emp_length'].apply(lambda x: 0 if x=='< 1 year' else(10 if x=='10+ years' else int(x[0])))
df['emp_length'].value_counts()


# ### issue_d column Treatment

# In[22]:


df['issue_d'].head()


# In[23]:


df['issue_month'] = df['issue_d'].apply(lambda x: x.split('-')[0])
df['issue_year'] = df['issue_d'].apply(lambda x : '20' + x.split('-')[1])
df.head()


# ### loan_status column Treatment

# In[24]:


# checking value count
df['loan_status'].value_counts()


# In[25]:


# The ones marked 'current' are neither fully paid not defaulted. So we cant predict defaulter for this 'Current' users
# Hence rows which has 'loan status - Current' are dropped
df = df[~(df['loan_status']=='Current')]
df.shape


# ## Data segmentation

# In[26]:


df.info()


# In[27]:


# data columns are segregated between continous and categorical

cont_cols = ['loan_amnt','funded_amnt','funded_amnt_inv','int_rate','installment','emp_length','annual_inc',
             'dti','pub_rec_bankruptcies']
cat_cols = ['term','grade','sub_grade','home_ownership','verification_status','loan_status','purpose','addr_state']


# In[28]:


df[cont_cols].describe()


# ## Outlier Detection and Treatment

# In[29]:


# outlier detection for continuous variables
for i in cont_cols:
    plt.title(i)
    sns.boxplot(df[i])
    plt.show()


# There is need of removing outlier for annual_inc column

# In[30]:


# finding out quartiles for annual inc column
df['annual_inc'].quantile([0.5,0.75,0.8,0.9,0.95,1])


# In[31]:


df = df[df['annual_inc'] < df['annual_inc'].quantile(0.95)]
df.shape


# Observations : 
# Though outlier is present, Data looks continuous for loan_amnt, funded_amnt, funded_amnt_inv, int_rate and installment columns.
# So need for Outlier treatment for this columns.
# However for 'annual_inc column' outlier treatment might be necessary, but not mandatory.
# In this analysis, No outlier treatment is done

# ### categorization of continous columns

# In[60]:


df['bin_loan_amnt'] = pd.cut(df['loan_amnt'], bins=7,precision =0,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k'])
df['bin_funded_amnt'] = pd.cut(df['funded_amnt'], bins=7,precision =0,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k'])
df['bin_funded_amnt_inv'] = pd.cut(df['funded_amnt_inv'], bins=7,precision =0,labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k'])
df['bin_installment'] = pd.cut(df['installment'], bins=14,precision =0,labels=['0-100','100-200','200-300','300-400','400-500','500-600','600-700','700-800','800--900','900-1000','1000-1100','1100-1200','1200-1300','1300-1400'])
df['bin_annual_inc'] = pd.cut(df['annual_inc'], bins=6,precision =0,labels=['0-1lac','1lac-2lac','2lac-3lac','3lac-4lac','4lac-5lac','5lac-6lac'])
df['bin_dti'] = pd.cut(df['dti'], bins=6,precision =0,labels=['0-5','5-10','10-15','15-20','20-25','25-30'])


# # 3) Data Analysis and Visualization

# ## Univariate Analysis

# In[33]:


# plotting loan_status Countplot

sns.countplot(df["loan_status"])
plt.show()


# ### Here we will be analyzing all categorical variables for loan_status = 'Charged off' 

# In[34]:


# Plotting countplot for 'term' for loan_status = 'Charged off' 

sns.countplot(df['term'], data = df[df['loan_status']=='Charged Off'])
plt.show()


# Observation : There are more defaulters for 36 months term

# In[35]:


# Plotting countplot for 'grade' for loan_status = 'Charged off' 

sns.countplot(df['grade'], data = df[df['loan_status']=='Charged Off'])
plt.show()


# Observation : There are more defaulters for 'B' grade Applicants

# In[36]:


# Plotting countplot for 'sub_grade' for loan_status = 'Charged off' 
fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['sub_grade'], data = df[df['loan_status']=='Charged Off'])
plt.xticks(rotation = 90)
plt.show()


# In[37]:


# Plotting grade for 
df['sub_grade2'] = df['sub_grade'].apply(lambda x: x[-1])
fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['grade'], order = ['A','B','C','D','E','F','G'],hue = df['sub_grade2'], data = df[df['loan_status']=='Charged Off'])
plt.show()


# Observation

# In[38]:


# Plotting countplot for 'home_ownership' for loan_status = 'Charged off' 

sns.countplot(df['home_ownership'], data = df[df['loan_status']=='Charged Off'])
plt.show()


# In[39]:


df['home_ownership'].value_counts()


# Observation : Applicants having home_ownership 'Rent' have high chances of loan defaulting

# In[40]:


# Plotting countplot for 'verification_status' for loan_status = 'Charged off' 

sns.countplot(df['verification_status'], data = df[df['loan_status']=='Charged Off'])
plt.show()


# Observation : Applicants having verification_status 'Not verified' have high chances of loan defaulting

# In[41]:


# Plotting countplot for 'purpose' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['purpose'], data = df[df['loan_status']=='Charged Off'])
plt.xticks(rotation = 90)
plt.show()


# Observation : Applicants having purpose 'debt consolidation' have high chances of loan defaulting

# In[42]:


# Plotting countplot for 'addr_state' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(15, 10))
sns.countplot(df['addr_state'], data = df[df['loan_status']=='Charged Off'])
plt.xticks(rotation = 90)
plt.show()


# Observation : Applicants having address state 'AZ' have high chances of loan defaulting

# In[43]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['issue_month'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# Observation : Loans issued in month 'december' have high chances of loan default

# In[44]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_loan_amnt'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# Observation : Loans issued for loan amount group 10k-20k have high chances of loan default

# In[45]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_funded_amnt'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# In[46]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_funded_amnt_inv'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# In[47]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_installment'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# In[48]:


# Plotting countplot for 'issue_month' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_annual_inc'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# In[62]:


# Plotting countplot for 'dti' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['bin_dti'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# In[63]:


# Plotting countplot for 'pub_rec_bankruptcies' for loan_status = 'Charged off' 

fig, ax = plt.subplots(figsize=(10, 7))
sns.countplot(df['pub_rec_bankruptcies'], data = df[df['loan_status']=='Charged Off'])

plt.xticks(rotation = 90)
plt.show()


# ## Insights

# In[ ]:





# ## Bivariate Analysis

# In[65]:


print(cont_cols,cat_cols)


# ### 1) Analyzing 'loan_amnt' variable against different categorical variables

# In[68]:


# Plotting loan amnt and term together
sns.barplot(y=df['loan_amnt'], x = df['term'], hue = df['loan_status'])
plt.show()


# Obaservation

# In[73]:


# Plotting loan amnt and grade together
plt.figure(figsize=(12,6))
sns.barplot(y=df['loan_amnt'], x = df['grade'], order = ['A','B','C','D','E','F','G'],hue = df['loan_status'])
plt.show()


# observation

# In[76]:


# Plotting loan amnt and home)ownership together
plt.figure(figsize=(8,6))
sns.barplot(y=df['loan_amnt'], x = df['home_ownership'],hue = df['loan_status'])
plt.show()


# Obseervation

# In[77]:


# Plotting loan amnt and verification_status together
plt.figure(figsize=(8,6))
sns.barplot(y=df['loan_amnt'], x = df['verification_status'],hue = df['loan_status'])
plt.show()


# Observation

# In[81]:


# Plotting loan amnt and purpose together
plt.figure(figsize=(12,8))
sns.barplot(y=df['loan_amnt'], x = df['purpose'],hue = df['loan_status'])
plt.xticks(rotation = 90)
plt.show()


# Observation

# In[82]:


# Plotting loan amnt and addr_state together
plt.figure(figsize=(12,8))
sns.barplot(y=df['loan_amnt'], x = df['addr_state'],hue = df['loan_status'])
plt.xticks(rotation = 90)
plt.show()


# Observation

# ### 2) Analyzing 'int_rate' variable against different categorical variables

# In[84]:


# Plotting int_rate and bin_loan_amnt together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['bin_loan_amnt'],hue = df['loan_status'])
plt.xticks(rotation = 90)
plt.show()


# In[86]:


# Plotting int_rate and purpose together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['purpose'],hue = df['loan_status'])
plt.xticks(rotation = 90)
plt.show()


# In[87]:


# Plotting int_rate and verification_status together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['verification_status'],hue = df['loan_status'])
plt.xticks(rotation = 90)
plt.show()


# In[90]:


# Plotting int_rate and grade together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['grade'],order = ['A','B','C','D','E','F','G'], hue = df['loan_status'])

plt.show()


# In[93]:


# Plotting int_rate and home_ownership together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['home_ownership'], hue = df['loan_status'])
plt.show()


# In[94]:


# Plotting int_rate and bin_annual_inc together
plt.figure(figsize=(12,8))
sns.barplot(y=df['int_rate'], x = df['bin_annual_inc'], hue = df['loan_status'])
plt.show()


# In[96]:


# Plotting bin_int_rate and dti together
plt.figure(figsize=(12,8))
sns.barplot(x=df['bin_int_rate'], y = df['dti'], hue = df['loan_status'])
plt.show()


# ### 3) Analyzing 'annual_inc' variable against different categorical variables

# In[98]:


# Plotting annual_inc and bin_loan_amnt  together
plt.figure(figsize=(8,6))
sns.barplot(x=df['bin_loan_amnt'], y = df['annual_inc'], hue = df['loan_status'])
plt.show()


# In[101]:


# Plotting annual_inc and purpose  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['purpose'], y = df['annual_inc'], hue = df['loan_status'])
plt.xticks(rotation=90)
plt.show()


# In[102]:


# Plotting annual_inc and home_ownership  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['home_ownership'], y = df['annual_inc'], hue = df['loan_status'])
plt.show()


# In[104]:


# Plotting annual_inc and verification_status  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['verification_status'], y = df['annual_inc'], hue = df['loan_status'])
plt.show()


# In[106]:


# Plotting annual_inc and verification_status  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['grade'], y = df['annual_inc'], order = ['A','B','C','D','E','F','G'], hue = df['loan_status'])
plt.show()


# ### 4) Analyzing other variable against different categorical variables

# In[107]:


# Plotting annual_inc and verification_status  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['bin_loan_amnt'], y = df['emp_length'], hue = df['loan_status'])
plt.show()


# In[108]:


# Plotting annual_inc and verification_status  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['bin_annual_inc'], y = df['emp_length'], hue = df['loan_status'])
plt.show()


# In[109]:


# Plotting annual_inc and verification_status  together
plt.figure(figsize=(12,8))
sns.barplot(x=df['home_ownership'], y = df['emp_length'], hue = df['loan_status'])
plt.show()


# In[ ]:




