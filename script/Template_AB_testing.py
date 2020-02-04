#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)


# In[2]:


df = pd.read_csv("../data/input/ab_data.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


# ユニークユーザのカウント
df["user_id"].nunique()


# In[5]:


# converted の平均
df.converted.mean()


# In[6]:


# 介入群が new_page に遷移していない行数
df[((df["group"] == "treatment") == (df["landing_page"] == "new_page")) == False].count()


# In[7]:


# 欠損値数
df.info()


# In[8]:


# 介入群!=new_page, 対照群!=old_page に対応する行インデックスを取得
i = df[((df["group"] == "treatment") == (df["landing_page"] == "new_page")) == False].index
i


# In[9]:


# 不当な行を削除
df2 = df.drop(i)
df2[((df2["group"] == "treatment") == (df2["landing_page"] == "new_page")) == False].shape[0]


# In[10]:


# ユニークユーザの数
df2["user_id"].nunique()


# In[11]:


# user_id の重複
df2[df2.duplicated(["user_id"], keep=False)]


# In[12]:


# 重複行削除
df2.drop_duplicates(subset="user_id", keep="first", inplace=True)


# In[13]:


# convert=1 の割合
(df2.query("converted == 1").converted.count()) / df2.shape[0]


# In[14]:


# 介入群のコンバージョン率
control_df = df2.query("group == 'control'")
Pold = control_df["converted"].mean()
Pold


# In[15]:


# 対照群のコンバージョン率
treatment_df = df2.query("group == 'treatment'")
Pnew = treatment_df["converted"].mean()
Pnew


# In[16]:


# 介入群の割合
df2.query("landing_page == 'new_page'").landing_page.count() / df2.shape[0]


# In[17]:


# クロス集計
crossed = pd.crosstab(df2.group, df2.converted)
crossed


# In[18]:


# $\Chi^2$-検定
from scipy.stats import chi2_contingency
x2, p, dof, expected = chi2_contingency(crossed)


# In[19]:


print("$Chi^2$-値: {}".format(x2))
print("確率: {}".format(p))
print("自由度: {}".format(dof))
print("expected: {}".format(expected))


# In[20]:


if p < 0.05:
    print("有意差があります")
else:
    print("有意差がありません")

