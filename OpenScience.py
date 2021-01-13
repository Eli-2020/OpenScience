#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Installing Pingouin package on Anaconda for conducting ANCOVA (if any)
# conda install -c conda-forge pingouin


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pingouin import ancova
import seaborn as sns

# Upload the csv file on JupyterLab before run the script
responses = pd.read_csv('responses.csv')


# In[ ]:


# Anonymizing mail adresses in the dataset
responses['Username'] = responses['Username'].apply(lambda x: 'respondent' + str(responses.loc[responses['Username'] == x].index[0]))

# Saving anonymized data set as a csv file
responses.to_csv('anonymous_responces.csv')

# Overviewing data
#responses.sample(5)


# In[ ]:


# Extracting pre- and post-test results and choice of learning method
pre = responses.iloc[0:, [1, 3, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 54, 57]]
post = responses.iloc[0:, [1, 3, 6, 9, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97, 54, 57]]

#pre.sample(5)
#post.sample(5)


# In[ ]:


# Renaming index to simplify
titles_pre = ['Respondent', 'Age', 'Degree', 'Relevant study', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Video', 'Text']
titles_post = ['Respondent', 'Age', 'Degree', 'Relevant study', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Video', 'Text']
pre.columns = titles_pre
post.columns = titles_post

#pre.head()
#post.head()


# In[ ]:


# Checking if null data is included in socres
print(pre.isnull().sum())
print('\n')
print(post.isnull().sum())


# In[ ]:


# Replacing "1.00 / 1" to 1, "0.00 / 1" to 0
pre_one = pre.mask(pre == '1.00 / 1', 1)
pre_scores = pre_one.mask(pre == '0.00 / 1', 0)

post_one = post.mask(post == '1.00 / 1', 1)
post_scores = post_one.mask(post == '0.00 / 1', 0)

#post_one = post.iloc[4:, :] == '1.00 / 1'
#post_one_zero = post_one.where(post_one != True, 1)
#post_scores = post_one_zero.where(post_one_zero != False, 0)

#pre_scores.sample(5)
#post_scores.sample(5)


# In[ ]:


# Replacing "Yes" in Video as video, "Yes!" in Text as text
v_t = pre['Video']
v_t.columns = ['Material']
v_t = v_t.replace('Yes', 'Video')
v_t = v_t.fillna('Text')

pre_scores['Material'] = v_t
post_scores['Material'] = v_t

#pre_scores.sample(5)
#post_scores.sample(5)


# In[ ]:


# Extracting video learners' results
video_gr_pre = pre_scores[pre_scores['Material'] == 'Video']
video_gr_post = post_scores[post_scores['Material'] == 'Video']

print(video_gr_pre)
print('\n')
print(video_gr_post)


# In[ ]:


# Extracting text learners' results
text_gr_pre = pre_scores[pre_scores['Material'] == 'Text']
text_gr_post = post_scores[pre_scores['Material'] == 'Text']

print(text_gr_pre)
print('\n')
print(text_gr_post)


# In[ ]:


# Visualizing age distribution with  pi chart
# Counting the number of each group
age_gr = pre_scores.groupby('Age').size()
#age_gr.head()

# Drawing a pi chart
label_age = ['18-25', '26-35', '46-']
plt.title('Age distribution')
plt.pie(age_gr, labels=label_age, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[ ]:


# Visualizing degree level distribution with  pi chart
# Counting the number of degree group
degree_gr = pre_scores.groupby('Degree').size()

# Drawing a pi chart
label_degree = ['Bachelor', 'Master', 'PhD']
plt.title('Degree distribution')
plt.pie(degree_gr, labels=label_degree, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[ ]:


# Visualizing having relevant background with  pi chart
# Counting the number of yes/no group
relevance = pre_scores.groupby('Relevant study').size()
print(relevance)

# Drawing a pi chart
relevance_degree = ['No', 'Yes']
plt.title('Studied public health, medicine, or virology')
plt.pie(relevance, labels=relevance_degree, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[ ]:


# Visualizing assigned learning material with pi chart
# Counting the number of yes/no group
material_assign = pre_scores.groupby('Material').size()
#print(material_assign)

#video_gr = pre_scores['material'].value_counts()['video']
#text_gr = pre_scores['material'].value_counts()['text']
#material_assign = [video_gr, text_gr]

# Drawing a pi chart
assign = ['Text-based', 'Video-based']
plt.title('Assigned learning material')
plt.pie(material_assign, labels=assign, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.show()


# In[ ]:


# Visualizing pre-post scores
# Each participants' score
pre_score_sum = pre_scores.iloc[:, 4:17].sum(axis=1)
post_score_sum = post_scores.iloc[:, 4:17].sum(axis=1)

labels_hist = ['pre-test', 'post-test']
plt.hist([pre_score_sum, post_score_sum], label=labels_hist)
plt.title("Score distribution of pre- and post-test")
plt.legend()
plt.show()


# In[ ]:


# ANCOVA
from pingouin import ancova
ancova_data = pd.DataFrame({'Material': v_t,
                       'Pre': pre_score_sum,
                       'Post': post_score_sum})
print(ancova_data)
print('\n' + 'Learn with video')
print(ancova_data[ancova_data['Material'] == 'Video'].describe().round(2))
print('\n'  + 'Learn with text')
print(ancova_data[ancova_data['Material'] == 'Text'].describe().round(2))
ancova(data=ancova_data, dv='Post', covar='Pre', between='Material')

#From the ANCOVA table we see that the p-value (p-unc = “uncorrected p-value”) for 'material' is 0.133439. 
#Since this value is more than 0.05, we cannot reject the null hypothesis (H0) that states 
#"There is no deference of performance in between two groups".
#Ref. https://pingouin-stats.org/generated/pingouin.ancova.html#pingouin.ancova


# In[ ]:


time_pre = pd.DataFrame({'Pre': np.repeat('Pre', 20)})
time_post = pd.DataFrame({'Post': np.repeat('Post', 20)})
seaborn_data = pd.DataFrame({'Material': ancova_data['Material'].append(ancova_data['Material'], ignore_index=True),
                        'Scores': ancova_data['Pre'].append(ancova_data['Post'], ignore_index=True),
                        'Time': time_pre['Pre'].append(time_post['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=seaborn_data , x='Time', y='Scores', hue='Material', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind')
plt.title('Chenges of average scores')
plt.show()
#Ref. https://seaborn.pydata.org/generated/seaborn.pointplot.html

