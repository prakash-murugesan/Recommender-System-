
# coding: utf-8

# # Reccommendation System

# ### Scenario

# You joined as the new memeber of a small start-up team. Together we are building a new app to sell cool 3rd party products! So far, the sales team worked tirelessly and managed to acquire over 50 merchants who each have different brands and products offerings. The developers have made a ton of progress on the mobile app and there's a bunch of user activity on the app. The analytics platform Mixpanel is used to collect all the user event data. The next step is to optimize user conversion rates by offering new recommendations based on the analytics data.

# ### Goal

# Your task is to recommend a product that a user is most likely to buy next using the purchase history provided.
# 
# For the purpose of this interview test, we have provided mock data on customer purchase history from an e-commerce retail company. The 'Purchased Product' events were queried through Mixpanel's API and is exported into the file training_mixpanel.txt, as provided, in JSON format. Each event describes a singular product purchased by a particular user, with descriptive attributes of the product (e.g., quantity, unit price). Transactions purchasing multiple products is denoted by invoice_no.

# ### Methodology 

# The pipeline first consisted of importing and exploring the data. This provided a vital intuition about the dataset and helped define the models and method of approaching this problem. The simplest and most robust model to go with is the user based collaborative filter recommender system. However, we can see that the data contains about 300 users or 10% of the population that have bought less than 5 unique items. This lends to a cold-start problem. Something that's taken care of by content-based recommendation systems. So, it was decided that the best method to tackle this task is by utilizing a **hybridized version of the collaborative filter and content based recommender systems** 

# ## 0. Import Libraries

# In[109]:


import numpy as np
import pandas as pd
import scipy
import math
import random
import json
import nltk 
import sklearn

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.corpus import stopwords
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


# ## 1. Data Exploration

# In[4]:


filename = "training_mixpanel.txt"

with open(filename) as f: 
    python_obj = json.load(f)
python_obj


# So we can see that the data set is a collection of events, all of them are a product being purchased. The event has properties such as invoice number, product id, description, quantity, invoice date, unit price, customer id, and country. 
# 
# The customer id is our user. We can manipulate this data set in a variety of ways to gain some useful information on how best to approach this problem. 
# 
# How often was a product bought? 
# How many different products did each user buy?
# Where were most of the products bought from? 
# what time of day were the products brought at?
# 
# All these may seem unrelated but can produce meaningful insights down the road. 

# In[3]:


print (json.dumps(python_obj[0], sort_keys=True, indent=4)) #pretty printing


# ### We move over to pandas to visualize and manipulate our data

# Since all the events are "purchased product" we can collect them separatel and work with just the properties

# In[5]:


properties = []
for i in range(len(python_obj)):
    properties.append(python_obj[i].get('properties'))


# In[84]:


prop = pd.DataFrame(properties)  #Creating a DataFrame for our data
prop


# In[85]:


print(len(pd.value_counts(prop["customer_id"])))


# In[86]:


print(len(pd.value_counts(prop["product_id"])))


# 4636 unique users and 3677 unique products

# In[87]:


prop.describe()


# highest number of purchases is 8000 and so is the highest number of returns. Smells fishy! Was it the same user? Was it the same product? Should we remove them as an outlier?

#  

# In[88]:


product_purchases = prop.groupby(['product_id'])['quantity'].sum().sort_values(ascending = False).reset_index()
product_purchases #absolute times products been purchased


# In[89]:


user_purchases = prop.groupby(['customer_id'])['quantity'].sum().sort_values(ascending = False).reset_index()
user_purchases #absolute times user has purchased products  i. purchases - returns


# Let's just keep it simple for now and look at solely the purchases for our recommendations. If they bought it in the first place they must have been looking for something similar. 

# In[90]:


user_purchases = prop.groupby(['customer_id', 'quantity']).size().sort_values(ascending = True).reset_index()
user_purchases #absolute times user has purchased products  i. purchases - returns


# User 12346 has bought and returned 74 thousand items..

# ## 2. Data Preprocessing

# Let's just keep it simple for now and look at solely the purchases for our recommendations. If they bought it in the first place they must have been looking for something similar. 

# In[91]:


prop = prop[prop['quantity']>0] #removes negative values i.e. returns
prop.head()


# In[92]:


print(len(pd.value_counts(prop["customer_id"])))


# In[93]:


print(len(pd.value_counts(prop["product_id"])))


#  Let's give a weight to these purchases by multiplying the quantity with unit price to get a weighted strength score. We might want to give good recommendations of most products bought but we also want to increase our revenue! 

# In[94]:


value = prop.quantity * prop.unit_price
#Scale our results with log so the data set isn't skewewd by a few large purchases
value_scaled = np.log2(1+value)  ## add one so that values below 1 don't push us back negative and zero values don't put as -inf
prop['value'] = value_scaled
#prop['quantityScaled'] = np.log(prop.quantity) #Perhaps nominal quantity is a better measure without economics playing a part
prop


# In[95]:


prop = prop.drop_duplicates() #let's drop duplicate rows
prop.describe()


# There were probably a few interactions that were only returns. This DF doesn't contain that.

# In[98]:


Purchases_df = prop.groupby(['customer_id', 'product_id'])['value'].mean().reset_index()
print('# of unique customer/item purchases: %d' % len(Purchases_df))
Purchases_df.head(10)


# In[99]:


Purchases_df.describe()


# ## 2. Data Pre-processing

# First thing to always do in the ML preprocessing pipeline is to split the dataset. 
# 
# If this step is done later on, we would have applied our model over the whole dataset and fit really well on it and would be testing ourselves again on the evaluated data. Like asking darth vader if hes evil and he says no hes not evil so we believe him. 

# In[ ]:


y = Purchases_df['customer_id']
y.iloc[:].value_counts()


# In[ ]:



training_purchases, testing_purchases = train_test_split(Purchases_df, stratify=y.iloc[:], test_size=0.2)
                                                           
                                                             

print('# of training purchases: %d' % len(training_purchases))
print('# of testing purchases: %d' % len(testing_purchases))


# In[ ]:


print(len(pd.value_counts(training_purchases["customer_id"])))


# In[ ]:


training_purchases, testing_purchases = train_test_split(prop, test_size = 0.2)  

print('# of training purchases: %d' % len(training_purchases))
print('# of testing purchases: %d' % len(testing_purchases))


# In[ ]:


print(len(pd.value_counts(testing_purchases["customer_id"])))


# In[100]:


#Indexing by personId to speed up the searches during evaluation
purchases_full_indexed_df = Purchases_df.set_index('customer_id')
#training_indexed_df = training_purchases.set_index('customer_id')
#testing_indexed_df = testing_purchases.set_index('customer_id')


# In[101]:


purchases_full_indexed_df


# ## 3. Content Based Filtering

# In[102]:


print(len(pd.value_counts(prop["product_id"])))


# In[103]:


print(len(pd.value_counts(prop["description"])))


# There seems to be 3659 unique product ids, but 3860 unique product descriptions. 
# 
# So there must be a product id with two different descriptions. We'll just focus on our product ids and discard the extra discriptions.

# In[104]:


collab_df = prop.drop_duplicates('product_id')
len(collab_df)
collab_df.head()


# In[105]:


from sklearn.metrics.pairwise import linear_kernel
#color = (['WHITE', 'BLACK', 'BLUE', "RED", 'GREEN', 'YELLOW', 'ORANGE', 'PURPLE', 'INDIGO'])
stopwords_list = stopwords.words('english') 

vectorizer = TfidfVectorizer(analyzer = 'word',
                           ngram_range=(1,2),
                           min_df=0.0003,
                           max_df=0.5,
                           max_features = None,
                           lowercase=True,
                           stop_words=stopwords_list)
item_ids = prop['product_id'].drop_duplicates().tolist()
tfidf_matrix = vectorizer.fit_transform(collab_df['description'])
tfidf_feature_names = vectorizer.get_feature_names()
#cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
tfidf_matrix


# Let's have a look at our features:

# In[106]:


tfidf_feature_names


# Now, to model the user profile, we take all the item profiles the user has bought and average them. 

# In[107]:


def get_item_profile(item_id):
    """Obtains a vector profiling an item"""
    print(item_id)
    idx = item_ids.index(item_id)
    #print(idx)
    item_profile = tfidf_matrix[idx:idx+1] #The 1000 element row vector for the item
    return item_profile

def get_item_profiles(ids):
    """Get all the item profiles for a given list of items"""
    if type(ids)==str:
        print (len(ids))
        ids = [ids] #if isinstance(ids, list) else [ids] #to make sure we are always dealing with lists, even when a single element
    
    print(len(ids))
    item_profiles_list = [get_item_profile(x) for x in ids] #get all the items
        
    item_profiles = scipy.sparse.vstack(item_profiles_list) #stack them on top of each other
    return item_profiles

def build_users_profile(customer_id, purchases_full_indexed_df): 
    """Get all the"""
    purchases_customer_df = purchases_full_indexed_df.loc[customer_id] #Receive the customer id and log all their purchases
    user_item_profiles = get_item_profiles(purchases_customer_df['product_id'])
    user_profile_norm = sklearn.preprocessing.normalize(user_item_profiles)
    #print(user_item_profiles)
    return user_item_profiles

def build_users_profiles():
    #purchases_full_indexed_df = Purchases_df.set_index('customer_id')
    user_profiles = {}
    for user_id in purchases_full_indexed_df.index.unique():
        user_profiles[user_id] = build_users_profile(user_id,purchases_full_indexed_df)
        
    return user_profiles


# In[110]:


user_profiles = build_users_profiles()
#user_profiles


# In[111]:


len(user_profiles)


# In[112]:


print(user_profiles)


# In[113]:


print(len(purchases_full_indexed_df.index.unique()))


# In[114]:


#Creating the mxn matrix
users = purchases_full_indexed_df.index.unique()
result = []
count=0
len(users)
for user_id in users:
    print("working on user #{}".format(user_id))
    user_product_similarity = cosine_similarity(user_profiles[user_id],tfidf_matrix)
    user_product_similarity = user_product_similarity.mean(axis=0)
    if count == 0:
        result = scipy.sparse.coo_matrix(user_product_similarity)
    else:
        result = scipy.sparse.vstack((result,user_product_similarity))
    count += 1


# In[115]:


result.shape


# In[116]:


content_based_result = result.todense()
content_based_result.shape


# In[117]:


content = np.array(content_based_result)
content


# In[201]:


np.savetxt("content_filter.csv", content, delimiter=",")


# ## 4. Collaborative Filtering

# There are many model-based CF algorithms to recommend items to users such as neural networks, bayerian networks, clustering models, and latent factor models such as Singular Value Decomposition (SVD) and, probabilistic latent semantic analysis. 

# ### Matrix Factorization: SVD

# Latent factor models compress the user-item matrix into a low-dimensional representation in terms of latent factors. One advantage of using this approach is that instead of having a high dimensional matrix containing a large number of missing values, we will be dealing with a much smaller matrix in lower-dimensional space. 
# 
# We will be using a popular latent factor model named Singular Value Decomposition (SVD). There are other frameworks such as surprise, mrec, python-recsys that are specific to Collaborative Filters, however, we'll stick with SciPy.

# In[118]:


Purchases_df = prop.groupby(['customer_id','product_id'])['value'].mean().reset_index()
print('# of unique customer/item purchases: %d' % len(Purchases_df))
Purchases_df.describe()


# In[119]:


#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = Purchases_df.pivot(index='customer_id',
                                                      columns='product_id',
                                                      values='value').fillna(0)
users_items_pivot_matrix_df.head(100)


# In[ ]:


Purchases_training_df = training_purchases.groupby(['customer_id','product_id'])['value'].mean().reset_index()
print('# of unique customer/item purchases: %d' % len(Purchases_df))
Purchases_df.describe()


# In[ ]:


Purchases_testing_df = testing_purchases.groupby(['customer_id','product_id'])['value'].mean().reset_index()
print('# of unique customer/item purchases: %d' % len(Purchases_df))
Purchases_df.describe()


# In[ ]:


#Creating a sparse pivot table with users in rows and items in columns

training_data_matrix_df = Purchases_training_df.pivot(index='customer_id',
                                                      columns='product_id',
                                                      values='value').fillna(0)
training_data_matrix_df.head(100)


# In[ ]:


#Creating a sparse pivot table with users in rows and items in columns

test_data_matrix = Purchases_testing_df.pivot(index='customer_id',
                                                      columns='product_id',
                                                      values='value').fillna(0)
test_data_matrix.head(100)


# In[120]:


users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_items_pivot_matrix[:10]


# In[121]:


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


# The number of matrix factors determines the number of features our model learns. The greater the number of features the better trained the model. However, it may then be prone to overfitting and a poor predictor of future events. 

# In[122]:


matrix_factors = 25 
U, sigma, Vt = svds(users_items_pivot_matrix, k = matrix_factors)


# In[123]:


U.shape


# In[124]:


Vt.shape


# In[125]:


sigma = np.diag(sigma)
sigma.shape


# In[126]:


all_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt)
all_user_predicted_ratings


# In[127]:


all_user_predicted_ratings.shape


# In[129]:


cf_preds_df = pd.DataFrame(all_user_predicted_ratings, 
                           columns = users_items_pivot_matrix_df.columns,
                           index=users_ids).transpose()
cf_preds_df.head(100)


# In[130]:


from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import label_binarize
#users_items_pivot_matrix = label_binarize(users_items_pivot_matrix, classes=[0, 1])
#all_user_predicted_ratings = label_binarize(all_user_predicted_ratings,classes=[0,1])
print (mean_squared_error(users_items_pivot_matrix,all_user_predicted_ratings))


# In[131]:


print (mean_squared_error(users_items_pivot_matrix,content_based_result))


# We can see that mean squared error wise the collaborative filter actually performs better than the content based filter, as can be expected for most cases in our dataset

# In[139]:


np.savetxt("collab_filter.csv", all_user_predicted_ratings, delimiter=",")


# ## 5. Hybrid content based and collaborative recommendation

# We can just smash the two matrix together by mutliplying them. However, we know that content based reccommdations are better equiped to overcome the cold start problem. So one method we could undertake is to weight the content based filter to have heavier implications for users that have bought less than 5 unique purchases. 

# In[136]:


hybrid = np.multiply(all_user_predicted_ratings,content)


# In[137]:


hybrid.shape


# In[138]:


print (mean_squared_error(users_items_pivot_matrix,hybrid))


# **HULK SMASH** doesn't work because most of the elements are zero for the content based matrix

# In[ ]:


all_user_predicted_ratings


# In[ ]:


content


# In[186]:


user_purchases = prop.groupby('customer_id')['quantity'].size().sort_values(ascending=False).reset_index()


# In[134]:


users_low_purchases = user_purchases[user_purchases.quantity<5]
users_low_purchases.describe()


# 291 users with purchases less than 5. These can be great candidates for content based recommendations

# In[140]:


users_low_purchases.head()


# In[141]:


check = cf_preds_df.transpose()
check


# In[142]:


check.index.values


# In[143]:


hybrid_matrix = all_user_predicted_ratings
row_number = 0
count = 0
for i in check.index.values:
    for customer_id in users_low_purchases['customer_id']:
        if i == customer_id:
            hybrid_matrix[row_number] = content_based_result[row_number]
            count +=1
            print(hybrid_matrix[row_number])
    row_number += 1


# In[144]:


print(count)


# In[145]:


hybrid_matrix.shape


# In[146]:


print (mean_squared_error(users_items_pivot_matrix,hybrid_matrix))


# In[198]:


np.savetxt("hybrid_filter.csv", hybrid_matrix, delimiter=",")


# ## Conclusion

# There is large room for improvements of the results. 
# 
# Firstly, I would have loved to tune the hyperparameters for both models with a cross validation set. With time considerations, I've just gone with best practices. 
# 
# Also, we've completely ignored the time, we could further leverage the available contextual information to model users preferences across time (period of day, day of week, month) and country.
# This contextual information can be easily incorporated in Learn-to-Rank models (like XGBoost Gradient Boosting Decision Trees with ranking objective) or Logistic models (with categorical features One-Hot encoded or Feature Hashed). 
# 
# There are more advanced techniques in RecSys research community that could be worth exploring, specially advanced Matrix Factorization and Deep Learning models.
