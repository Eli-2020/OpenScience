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
raw_responses = pd.read_csv('responses.csv')


# In[ ]:


responses = raw_responses.drop('Do you have any recommendations or feedback on the experiment?', axis=1)


# In[ ]:


# Anonymizing mail adresses in the dataset
responses['Username'] = responses['Username'].apply(lambda x: 'respondent' + str(responses.loc[responses['Username'] == x].index[0]))

# Saving anonymized data set as a csv file
responses.to_csv('anonymous_responces.csv')

# Overviewing data
responses.sample(5)


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
#print(age_gr)

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
#print(relevance)

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
import matplotlib.ticker as ticker

pre_score_sum_t = pre_scores.groupby('Material').get_group('Text')
pre_score_sum_t = pre_score_sum_t.iloc[:, 4:17].sum(axis=1)

pre_score_sum_v = pre_scores.groupby('Material').get_group('Video')
pre_score_sum_v = pre_score_sum_v.iloc[:, 4:17].sum(axis=1)

post_score_sum_t = post_scores.groupby('Material').get_group('Text')
post_score_sum_t = post_score_sum_t.iloc[:, 4:17].sum(axis=1)

post_score_sum_v = post_scores.groupby('Material').get_group('Video')
post_score_sum_v = post_score_sum_v.iloc[:, 4:17].sum(axis=1)

labels_hist = ['pre-test (text)', 'pre-test (video)', 'post-test (text)', 'post-test (video)']
plt.figure(figsize=(7, 5))
plt.hist([pre_score_sum_t, pre_score_sum_v, post_score_sum_t, post_score_sum_v], label=labels_hist, color=['#87CEFA', '#F4A460', '#1E90FF', '#D2691E'])
plt.title("Score distribution of pre- and post-test")
plt.xlabel('Score')
plt.ylabel('Count')
plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.show()


# In[ ]:


# Two-sided Mann-Whitney U Test
from pingouin import mwu

pre_score_sum = pre_scores.iloc[:, 4:17].sum(axis=1)
post_score_sum = post_scores.iloc[:, 4:17].sum(axis=1)

# Comparing Pre and Post tests including both learning styles
print('Pre and Post tests including both learning styles')
mwu(post_score_sum, pre_score_sum, tail='two-sided')

#ref: https://pingouin-stats.org/generated/pingouin.mwu.html#rf5915ba8ddc9-2
#ref: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test


# In[ ]:


# Two-sided Mann-Whitney U Test
# Comparing Pre and Post tests in terms of text-based learning
print('Pre and Post tests in terms of text-based learning')
mwu(post_score_sum_t, pre_score_sum_t, tail='two-sided')


# In[ ]:


# Two-sided Mann-Whitney U Test
# Comparing Pre and Post tests in terms of video-based learning
print('Pre and Post tests in terms of video-based learning')
mwu(post_score_sum_v, pre_score_sum_v, tail='two-sided')


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


# Visualizing changes of average scores
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


# In[ ]:


# Data analysis by groups
# Insights of respondents in terms of text and video group (Age)
age_gr_text = pre_scores.groupby('Material').get_group('Text')
age_gr_text = age_gr_text.groupby('Age').size()

age_gr_video = pre_scores.groupby('Material').get_group('Video')
age_gr_video = age_gr_video.groupby('Age').size()

age_groupby = pd.DataFrame({'Text': age_gr_text, 'Video': age_gr_video})
age_groupby = age_groupby.fillna(0).round().astype(int)
#print(age_groupby)

# Drawing a bar chart
fig, ax = plt.subplots(figsize=(6, 6))
age_groupby.T.plot(kind='bar', stacked=True, ax=ax)
plt.title('Group by age')
plt.show()


# In[ ]:


# Insights of respondents in terms of text and video group (Degree)
degree_gr_text = pre_scores.groupby('Material').get_group('Text')
degree_gr_text = degree_gr_text.groupby('Degree').size()

degree_gr_video = pre_scores.groupby('Material').get_group('Video')
degree_gr_video = degree_gr_video.groupby('Degree').size()

degree_groupby = pd.DataFrame({'Text': degree_gr_text, 'Video': degree_gr_video})
degree_groupby = degree_groupby.fillna(0).round().astype(int)
#print(degree_groupby)

# Drawing a bar chart
fig, ax = plt.subplots(figsize=(6, 6))
degree_groupby.T.plot(kind='bar', stacked=True, ax=ax)
plt.title('Group by degree')
plt.show()


# In[ ]:


# Insights of respondents in terms of text and video group (Relevant study)
bg_gr_text = pre_scores.groupby('Material').get_group('Text')
bg_gr_text = bg_gr_text.groupby('Relevant study').size()

bg_gr_video = pre_scores.groupby('Material').get_group('Video')
bg_gr_video = bg_gr_video.groupby('Relevant study').size()

bg_groupby = pd.DataFrame({'Text': bg_gr_text, 'Video': bg_gr_video})
bg_groupby = bg_groupby.fillna(0).round().astype(int)
#print(bg_groupby)

# Drawing a bar chart
fig, ax = plt.subplots(figsize=(6, 6))
bg_groupby.T.plot(kind='bar', stacked=True, ax=ax)
plt.title('Group by relevant study')
plt.show()


# In[ ]:


# Summery & visualization of scores by age (text group)
text_age_gr_pre = text_gr_pre.copy()
text_age_gr_pre['Pre score'] = text_age_gr_pre.iloc[:, 4:17].sum(axis=1)

text_age_gr_post = text_gr_post.copy()
text_age_gr_post['Post score'] = text_age_gr_post.iloc[:, 4:17].sum(axis=1)

summary_t1 = text_age_gr_pre.groupby('Age').get_group('18 to 25 years old').describe().round(2)
summary_t1['Post score'] = text_age_gr_post.groupby('Age').get_group('18 to 25 years old').describe().round(2)

summary_t2 = text_age_gr_pre.groupby('Age').get_group('26 to 35 years old').describe().round(2)
summary_t2['Post score'] = text_age_gr_post.groupby('Age').get_group('26 to 35 years old').describe().round(2)

print('Pre/Post scores by age group: 18-25 (Text)')
print(summary_t1)
print('\n')
print('Pre/Post scores by age group: 26-35 (Text)')
print(summary_t2)
print('\n')

pre_by_age_t = pd.DataFrame({'Pre': np.repeat('Pre', len(text_age_gr_pre.index))})
post_by_age_t = pd.DataFrame({'Post': np.repeat('Post', len(text_age_gr_post.index))})
text_by_age = pd.DataFrame({'Age': text_age_gr_pre['Age'].append(text_age_gr_post['Age'], ignore_index=True),
                        'Scores': text_age_gr_pre['Pre score'].append(text_age_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_age_t['Pre'].append(post_by_age_t['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=text_by_age, x='Time', y='Scores', hue='Age', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['18 to 25 years old', '26 to 35 years old'])
plt.title('Chenges of average scores by age group (Text)')
plt.show()


# In[ ]:


# Summery & visualization of scores by age (video group)
video_age_gr_pre = video_gr_pre.copy()
video_age_gr_pre['Pre score'] = video_age_gr_pre.iloc[:, 4:17].sum(axis=1)

video_age_gr_post = video_gr_post.copy()
video_age_gr_post['Post score'] = video_age_gr_post.iloc[:, 4:17].sum(axis=1)

summary_v1 = video_age_gr_pre.groupby('Age').get_group('18 to 25 years old').describe().round(2)
summary_v1['Post score'] = video_age_gr_post.groupby('Age').get_group('18 to 25 years old').describe().round(2)
summary_v2 = video_age_gr_pre.groupby('Age').get_group('26 to 35 years old').describe().round(2)
summary_v2['Post score'] = video_age_gr_post.groupby('Age').get_group('26 to 35 years old').describe().round(2)
summary_v3 = video_age_gr_pre.groupby('Age').get_group('More than 45 years old').describe().round(2)
summary_v3['Post score'] = video_age_gr_post.groupby('Age').get_group('More than 45 years old').describe().round(2)

print('Pre/Post scores by age group: 18-25 (Video)')
print(summary_v1)
print('\n')
print('Pre/Post scores by age group: 26-35 (Video)')
print(summary_v2)
print('\n')
print('Pre/Post scores by age group: Over 45 (Video)')
print(summary_v3)
print('\n')

pre_by_age_v = pd.DataFrame({'Pre': np.repeat('Pre', len(video_age_gr_pre.index))})
post_by_age_v = pd.DataFrame({'Post': np.repeat('Post', len(video_age_gr_post.index))})
video_by_age = pd.DataFrame({'Age': video_age_gr_pre['Age'].append(video_age_gr_post['Age'], ignore_index=True),
                        'Scores': video_age_gr_pre['Pre score'].append(video_age_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_age_v['Pre'].append(post_by_age_v['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=video_by_age, x='Time', y='Scores', hue='Age', dodge=True, markers=['o', 's', 'x'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['18 to 25 years old', '26 to 35 years old', 'More than 45 years old'])
plt.title('Chenges of average scores by age group (Video)')
plt.show()


# In[ ]:


# Summery & visualization of scores by degree (text group)
text_degree_gr_pre = text_gr_pre.copy()
text_degree_gr_pre['Pre score'] = text_degree_gr_pre.iloc[:, 4:17].sum(axis=1)

text_degree_gr_post = text_gr_post.copy()
text_degree_gr_post['Post score'] = text_degree_gr_post.iloc[:, 4:17].sum(axis=1)

summary_t3 = text_degree_gr_pre.groupby('Degree').get_group('Bachelor').describe().round(2)
summary_t3['Post score'] = text_degree_gr_post.groupby('Degree').get_group('Bachelor').describe().round(2)

summary_t4 = text_degree_gr_pre.groupby('Degree').get_group('Master').describe().round(2)
summary_t4['Post score'] = text_degree_gr_post.groupby('Degree').get_group('Master').describe().round(2)

summary_t5 = text_degree_gr_pre.groupby('Degree').get_group('PhD').describe().round(2)
summary_t5['Post score'] = text_degree_gr_post.groupby('Degree').get_group('PhD').describe().round(2)


print('Pre/Post scores by degree group: Bachelor (Text)')
print(summary_t3)
print('\n')
print('Pre/Post scores by degree group: Master (Text)')
print(summary_t4)
print('\n')
print('Pre/Post scores by degree group: PhD (Text)')
print(summary_t5)
print('\n')

pre_by_degree_t = pd.DataFrame({'Pre': np.repeat('Pre', len(text_degree_gr_pre.index))})
post_by_degree_t = pd.DataFrame({'Post': np.repeat('Post', len(text_degree_gr_post.index))})
text_by_degree = pd.DataFrame({'Degree': text_degree_gr_pre['Degree'].append(text_degree_gr_post['Degree'], ignore_index=True),
                        'Scores': text_degree_gr_pre['Pre score'].append(text_degree_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_degree_t['Pre'].append(post_by_degree_t['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=text_by_degree, x='Time', y='Scores', hue='Degree', dodge=True, markers=['o', 's', 'x'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['Bachelor', 'Master', 'PhD'])
plt.title('Chenges of average scores by degree group (Text)')
plt.show()


# In[ ]:


# Summery & visualization of scores by degree (video group)
video_degree_gr_pre = video_gr_pre.copy()
video_degree_gr_pre['Pre score'] = video_degree_gr_pre.iloc[:, 4:17].sum(axis=1)

video_degree_gr_post = video_gr_post.copy()
video_degree_gr_post['Post score'] = video_degree_gr_post.iloc[:, 4:17].sum(axis=1)

summary_v4 = video_degree_gr_pre.groupby('Degree').get_group('Bachelor').describe().round(2)
summary_v4['Post score'] = video_degree_gr_post.groupby('Degree').get_group('Bachelor').describe().round(2)

summary_v5 = video_degree_gr_pre.groupby('Degree').get_group('Master').describe().round(2)
summary_v5['Post score'] = video_degree_gr_post.groupby('Degree').get_group('Master').describe().round(2)


print('Pre/Post scores by degree group: Bachelor (Video)')
print(summary_v4)
print('\n')
print('Pre/Post scores by degree group: Master (Video)')
print(summary_v5)
print('\n')

pre_by_degree_v = pd.DataFrame({'Pre': np.repeat('Pre', len(video_degree_gr_pre.index))})
post_by_degree_v = pd.DataFrame({'Post': np.repeat('Post', len(video_degree_gr_post.index))})
video_by_degree = pd.DataFrame({'Degree': video_degree_gr_pre['Degree'].append(video_degree_gr_post['Degree'], ignore_index=True),
                        'Scores': video_degree_gr_pre['Pre score'].append(video_degree_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_degree_v['Pre'].append(post_by_degree_v['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=video_by_degree, x='Time', y='Scores', hue='Degree', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['Bachelor', 'Master'])
plt.title('Chenges of average scores by degree group (Video)')
plt.show()


# In[ ]:


# Summery & visualization of scores by relevant study (text group)
text_bg_gr_pre = text_gr_pre.copy()
text_bg_gr_pre['Pre score'] = text_bg_gr_pre.iloc[:, 4:17].sum(axis=1)

text_bg_gr_post = text_gr_post.copy()
text_bg_gr_post['Post score'] = text_bg_gr_post.iloc[:, 4:17].sum(axis=1)

summary_t6 = text_bg_gr_pre.groupby('Relevant study').get_group('Yes').describe().round(2)
summary_t6['Post score'] = text_bg_gr_post.groupby('Relevant study').get_group('Yes').describe().round(2)

summary_t7 = text_bg_gr_pre.groupby('Relevant study').get_group('No').describe().round(2)
summary_t7['Post score'] = text_bg_gr_post.groupby('Relevant study').get_group('No').describe().round(2)

print('Pre/Post scores by relevant study: Yes (Text)')
print(summary_t6)
print('\n')
print('Pre/Post scores by relevant study: No (Text)')
print(summary_t7)
print('\n')

pre_by_bg_t = pd.DataFrame({'Pre': np.repeat('Pre', len(text_bg_gr_pre.index))})
post_by_bg_t = pd.DataFrame({'Post': np.repeat('Post', len(text_bg_gr_post.index))})
text_by_bg = pd.DataFrame({'Relevant study': text_bg_gr_pre['Relevant study'].append(text_bg_gr_post['Relevant study'], ignore_index=True),
                        'Scores': text_bg_gr_pre['Pre score'].append(text_bg_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_bg_t['Pre'].append(post_by_bg_t['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=text_by_bg, x='Time', y='Scores', hue='Relevant study', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['Yes', 'No'])
plt.title('Chenges of average scores by relevant study group (Text)')
plt.show()


# In[ ]:


# Summery & visualization of scores by relevant study (video group)
video_bg_gr_pre = video_gr_pre.copy()
video_bg_gr_pre['Pre score'] = video_bg_gr_pre.iloc[:, 4:17].sum(axis=1)

video_bg_gr_post = video_gr_post.copy()
video_bg_gr_post['Post score'] = video_bg_gr_post.iloc[:, 4:17].sum(axis=1)

summary_v6 = video_bg_gr_pre.groupby('Relevant study').get_group('Yes').describe().round(2)
summary_v6['Post score'] = video_bg_gr_post.groupby('Relevant study').get_group('Yes').describe().round(2)

summary_v7 = video_bg_gr_pre.groupby('Relevant study').get_group('No').describe().round(2)
summary_v7['Post score'] = video_bg_gr_post.groupby('Relevant study').get_group('No').describe().round(2)

print('Pre/Post scores by relevant study: Yes (Video)')
print(summary_v6)
print('\n')
print('Pre/Post scores by relevant study: No (Video)')
print(summary_v7)
print('\n')

pre_by_bg_v = pd.DataFrame({'Pre': np.repeat('Pre', len(video_bg_gr_pre.index))})
post_by_bg_v = pd.DataFrame({'Post': np.repeat('Post', len(video_bg_gr_post.index))})
video_by_bg = pd.DataFrame({'Relevant study': video_bg_gr_pre['Relevant study'].append(video_bg_gr_post['Relevant study'], ignore_index=True),
                        'Scores': video_bg_gr_pre['Pre score'].append(video_bg_gr_post['Post score'], ignore_index=True),
                        'Time': pre_by_bg_v['Pre'].append(post_by_bg_v['Post'], ignore_index=True)})
sns.set()
sns.pointplot(data=video_by_bg, x='Time', y='Scores', hue='Relevant study', dodge=True, markers=['o', 's'],
  	      capsize=0.1, errwidth=0.5, palette='colorblind', hue_order = ['Yes', 'No'])
plt.title('Chenges of average scores by relevant study group (Video)')
plt.show()

