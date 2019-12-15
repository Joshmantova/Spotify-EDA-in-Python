#!/usr/bin/env python
# coding: utf-8

# In[205]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.style.use("fivethirtyeight")


# In[18]:


df = pd.read_csv("Downloads/top50.csv", encoding='latin1', index_col=0)


# In[19]:


df.head()


# #### What in the world is canadian pop??

# In[56]:


df[df['Genre'] == 'canadian pop']


# #### I guess canadian pop is not that big in the top 50.. Looks like only Shawn Mendes produces it. Let's see what else
# #### we're working with.

# In[21]:


df.columns


# #### What do histograms of the quantitative columns look like?

# In[135]:


df.hist();


# #### Let's use Matplotlib directly for this one as the Pandas histogram figure doesn't look very clear.

# In[136]:


fig, axs = plt.subplots(nrows=3,ncols=3, figsize=(10,10))
q_cols = ['Energy', 'Danceability', 'Acousticness..', 'Length.', 'Liveness', 'Loudness..dB..', 'Popularity', 'Speechiness.', 'Valence.']
for ax, col in zip(axs.flatten(), q_cols):
    ax.hist(df[col])
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show();


# #### Now we're talking.. This looks a lot more clear. It looks like most of the variables are heavily skewed either left or right.
# #### Only a few of the columns in here seem to have a normal distribution. That makes sense as they weren't randomly selected
# #### from the poulation; thus we wouldn't necessarily expect them to be normally distributed nor would we expect them to 
# #### be representative of all songs. Let's look a little deeper into Valence.

# In[69]:


val_m = df['Valence.'].mean()
print(f"The mean value of valence from this dataset was: {val_m}")
print(f"The range of values valence took on in this dataset was from {df['Valence.'].min()} to {df['Valence.'].max()}")


# #### It would appear that valence might be rated on a scale from 0 to 100, but we cannot be sure.
# #### The only thing we can tell here is that songs in this dataset took on valence values from 10 to 95.
# #### I wonder what kind of songs were high in valence and what kind of songs are low in valence..

# In[229]:


df.sort_values('Valence.', ascending=False).head(10).reset_index()


# In[228]:


df.sort_values('Valence.').head(10).reset_index()


# In[218]:


df_sortval = df.sort_values(by='Valence.', ascending=False).copy()
song_names = df_sortval['Track.Name'].head(10).values
artist_names = df_sortval['Artist.Name'].head(10).values
x = song_names + ' by ' + artist_names
x = x[::-1]
y = df_sortval['Valence.'].head(10)
y = y[::-1]
fig, ax = plt.subplots()

ax.barh(x, y)
ax.set_xlabel("Valence")
ax.set_ylabel("Song Names")
ax.set_title("Top 10 songs sorted by valence")
plt.show();


# In[219]:


gb = df.groupby('Artist.Name')[['Track.Name']].count().sort_values(by='Track.Name', ascending=False).head(10)


# In[222]:


y = gb.values.flatten()[::-1]
x = gb.index[::-1]


# In[227]:


fig, ax = plt.subplots()

ax.barh(x, y)
ax.set_xlabel("Number of songs in the top 50 list")
ax.set_ylabel("Artist name")
ax.set_title("Number of songs an artist has in the top 50 list")
plt.show();


# In[ ]:




