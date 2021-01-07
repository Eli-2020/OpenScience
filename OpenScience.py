#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Installing Pingouin package for conducting ANCOVA (if any)
# conda install -c conda-forge pingouin


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pingouin import ancova
import seaborn as sns

# Upload the csv file on JupyterLab before run the script
responses = pd.read_csv('responses.csv')


# In[ ]:


# Overviewing data
responses.sample(5)


# In[ ]:


# Extracting respondents' background info
respondents = responses.iloc[0:, [3, 6, 9]]
respondents.head()


# In[ ]:


# Extracting pre- and post-test results and choice of learning method
pre = responses.iloc[0:, [13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]]
post = responses.iloc[0:, [61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97]]
material = responses.iloc[0:, [54, 57]]
#post.sample(5)
#pre.sample(5)


# In[ ]:


# Renaming index to simplify
titles_pre = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']
titles_post = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']
titles_material = ['video', 'text']
pre.columns = titles_pre
post.columns = titles_post
material.columns = titles_material

#pre.head()
#post.head()
#material.sample(5)


# In[ ]:


# Checking if null data is included in socres
#print(pre.isnull().sum())
#print('\n')
#print(post.isnull().sum())
#print('\n')
#print(material.isnull().sum())


# In[ ]:


# Replacing "1.00 / 1" to 1, "0.00 / 1" to 0
pre_one = pre.iloc[:, :] == '1.00 / 1'
pre_one_zero = pre_one.where(pre_one != True, 1)
pre_scores = pre_one_zero.where(pre_one_zero != False, 0)

post_one = post.iloc[:, :] == '1.00 / 1'
post_one_zero = post_one.where(post_one != True, 1)
post_scores = post_one_zero.where(post_one_zero != False, 0)

#print(pre_scores)
#print('\n')
#print(post_scores)


# In[ ]:


# Replacing "Yes" in video as v, "Yes!" in text as t
v_t = material['video']
v_t.columns = ['material']
v_t = v_t.replace('Yes', 'video')
v_t = v_t.fillna('text')

pre_scores['material'] = v_t
post_scores['material'] = v_t

print(pre_scores)
print('\n')
print(post_scores)


# In[ ]:


# Extracting video learners' results
video_gr_pre = pre_scores[pre_scores['material'] == 'video']
video_gr_post = post_scores[post_scores['material'] == 'video']

print(video_gr_pre)
print('\n')
print(video_gr_post)


# In[ ]:


# Extracting text learners' results
text_gr_pre = pre_scores[pre_scores['material'] == 'text']
text_gr_post = post_scores[pre_scores['material'] == 'text']

print(text_gr_pre)
print('\n')
print(text_gr_post)


# In[34]:


# Visualizing age distribution with  pi chart
# Counting the number of each group
age_gr1 = respondents['Which age group do you belong to?'].value_counts()['18 to 25 years old']
age_gr2 = respondents['Which age group do you belong to?'].value_counts()['26 to 35 years old']

#no respondent fits this range 
#age_gr3 = respondents['Which age group do you belong to?'].value_counts()['36 to 45 years old']
age_gr4 = respondents['Which age group do you belong to?'].value_counts()['More than 45 years old']
age_gr = [age_gr1, age_gr2, age_gr4]

# Drawing a pi chart
label_age = ['18-25', '26-35', '46-']
plt.title('Age distribution')
plt.pie(age_gr, labels=label_age, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[35]:


# Visualizing degree level distribution with  pi chart
# Counting the number of degree group
degree_gr1 = respondents['What is your maximum educational level obtained?'].value_counts()['Bachelor']
degree_gr2 = respondents['What is your maximum educational level obtained?'].value_counts()['Master']
degree_gr3 = respondents['What is your maximum educational level obtained?'].value_counts()['PhD']
degree_gr = [degree_gr1, degree_gr2, degree_gr3]

# Drawing a pi chart
label_degree = ['Bachelor', 'Master', 'PhD']
plt.title('Degree distribution')
plt.pie(degree_gr, labels=label_degree, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[36]:


# Visualizing having relevant background with  pi chart
# Counting the number of yes/no group
relevant_gr = respondents['Is your educational background directly related to public health, medicine, or virology?'].value_counts()['Yes']
irrelevant_gr = respondents['Is your educational background directly related to public health, medicine, or virology?'].value_counts()['No']
relevance = [relevant_gr, irrelevant_gr]

# Drawing a pi chart
relevance_degree = ['Yes', 'No']
plt.title('Studied public health, medicine, or virology')
plt.pie(relevance, labels=relevance_degree, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[37]:


# Visualizing assigned learning material with pi chart
# Counting the number of yes/no group
video_gr = pre_scores['material'].value_counts()['video']
text_gr = pre_scores['material'].value_counts()['text']
material_assign = [video_gr, text_gr]

# Drawing a pi chart
assign = ['Video-based', 'Text-based']
plt.title('Assigned learning material')
plt.pie(material_assign, labels=assign, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[38]:


# Visualizing pre-post scores
# Each participants' score
pre_score_sum = pre_scores.iloc[:, :13].sum(axis=1)
post_score_sum = post_scores.iloc[:, :13].sum(axis=1)

labels_hist = ['pre-test', 'post-test']
plt.hist([pre_score_sum, post_score_sum], label=labels_hist)
plt.title("Score distribution of pre- and post-test")
plt.legend()
plt.show()


# In[39]:


# ANCOVA
from pingouin import ancova
ancova_data = pd.DataFrame({'material': v_t,
                       'pre': pre_score_sum,
                       'post': post_score_sum})
print(ancova_data)
print('\n')
print(ancova_data[ancova_data['material'] == 'video'].describe())
print('\n')
print(ancova_data[ancova_data['material'] == 'text'].describe())
ancova(data=ancova_data, dv='post', covar='pre', between='material')

#From the ANCOVA table we see that the p-value (p-unc = “uncorrected p-value”) for 'material' is 0.116775. 
#Since this value is more than 0.05, we cannot reject the null hypothesis (H0) that states 
#"There is no deference of performance in between two groups".
#Ref. https://pingouin-stats.org/generated/pingouin.ancova.html#pingouin.ancova


# In[ ]:


time_pre = pd.DataFrame({'Pre': np.repeat('Pre', 20)})
time_post = pd.DataFrame({'Post': np.repeat('Post', 20)})
seaborn_data = pd.DataFrame({'Material': ancova_data['material'].append(ancova_data['material'], ignore_index=True),
                        'Scores': ancova_data['pre'].append(ancova_data['post'], ignore_index=True),
                        'Time': time_pre['Pre'].append(time_post['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=seaborn_data , x='Time', y='Scores', hue='Material', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind')
plt.title('Chenges of average scores')
plt.show()
#Ref. https://seaborn.pydata.org/generated/seaborn.pointplot.html


# In[ ]:




