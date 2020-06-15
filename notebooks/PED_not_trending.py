# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 133} colab_type="code" id="RWSJpsyKqHjH" outputId="e65be45c-5485-45a9-ee9f-03e4df37f740"
import nltk
nltk.download("punkt")

import os

path = "../data/"
files = os.listdir(path)
files
# %% colab={"base_uri": "https://localhost:8080/", "height": 324} colab_type="code" id="DbzcWu-ZnRsQ" outputId="1fcb3627-9804-480a-aa0f-0c5253263725"
import pandas as pd

pd.set_option("colwidth", None)

df = pd.read_csv(path + "/" + "videos_not_trending.csv", index_col=0)
df = df.rename(columns={"description ": "description"})
print(df.shape)
df.head(3) 

# %% [markdown] colab_type="text" id="l8P8ev47QLxX"
# ## Unwanted attributes
# - We do not need to analyze **views**, **likes**, **dislikes** or **comment_count** as we cannot base the trending guidelines upon such statistics

# %% [markdown] colab_type="text" id="Yggnu6MUGqIE"
# ## Check for **missing values**
# Apart from category_id column about which we already know it has values missing, there are other attributes with missing data.
#
# ### Description

# %% colab={"base_uri": "https://localhost:8080/", "height": 33} colab_type="code" id="iIPd0TVvFqAG" outputId="181ade2b-8956-4db3-cc91-5842f68927d3"
missing_values_df = df.drop(["category_id"], axis=1)
missing_values_df = missing_values_df[missing_values_df.isnull().any(axis=1)]

for cname in missing_values_df.columns:
    check_nulls = missing_values_df[[cname]].isnull().sum().values[0]
    if check_nulls > 0:
        print("Missing values in column", cname, ":", check_nulls)

# %% [markdown] colab_type="text" id="S0qEV6gbO41G"
# There are NaNs in column `description`.
#
# **Solution**: Replace `NaN`s with "no description"

# %% colab={"base_uri": "https://localhost:8080/", "height": 117} colab_type="code" id="kYPvhOLo3ucr" outputId="6e416f75-ba35-4ce4-94b3-126a7645e3b1"
df.loc[df["description"].isna(), "description"] = "no description"

# %% [markdown] colab_type="text" id="0qhw5A3Z3tAE"
# ### Tags
# We can also observe that there can be missing tags, represented as `[none]`. We leave it as it is as no tags is also some kind of an information.

# %% colab={"base_uri": "https://localhost:8080/", "height": 33} colab_type="code" id="pqvqxS6kIxp4" outputId="084bc9f1-d850-4fc8-d64b-c787e273f9d0"
df[df["tags"] == "[none]"].shape

# %% [markdown] colab_type="text" id="Kl53OivLeka_"
#
# ### Video_id
#
# Some `video_ids` seem corrupted:
# > #NAZWA?

# %% colab={"base_uri": "https://localhost:8080/", "height": 154} colab_type="code" id="-oU8cukPeLH4" outputId="c206ae85-3aa7-48c0-a7d4-b07795836f12"
print(
    "Count #NAZWA?:",
    df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))]["video_id"].shape,
)
df[df["video_id"].apply(lambda x: any([not char.isalnum() and char not in "-_" for char in x]))][
    ["video_id", "title"]
].head(3)

# %% [markdown] colab_type="text" id="gW6EAgaOTOhe"
# ### Single video - different descriptions and titles ...
#
# Typical `video_id` contains alphanumeric characters, and `-` or `_`, and is 11 characters long.
#
# Grouping by `video_id`, and looking at columns of our interest there are some columns which values are different:
# - trending_date - as expected,
# - description - there are very slight differences between those (few characters diff), so it is definitely not enough to say that given the same title, the video is different. <u>It rather means that the user who uploaded the video, has decided to change a description a few days after the upload.</u>
# - tags - there are very rare cases, but also tags can vary!
# - title - unfortunately, in this column we can also observe changes.
#
# There is also `category_id` which yields two unique values when some of them are NaNs.
#
# After the analysis, we decided that the only column that can distinguish videos between themselves, aside from `video_id`, is `publish_time`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="koGb2t6cOYjA" outputId="e131f452-0377-4e65-d514-6b536f60ad09"
for example_video_id in df["video_id"].values[:5]:
    if "NAZWA" not in example_video_id:
        video_id_df = df[df["video_id"] == example_video_id]
        for cname in video_id_df.columns:
            if cname not in ["category_id", "views", "likes", "dislikes", "comment_count"]:
                count_unique = len(video_id_df[cname].unique())
                if count_unique > 1:
                    if cname == "title" or cname == "tags" or cname == "description":
                        print("\nnumber of unique '", cname, "': ", count_unique, '\n')
                        print(video_id_df[cname].unique())

# %% [markdown] colab_type="text" id="WmM9S738XqSh"
# > We can replace "#NAZWA?" with manually-generated video_ids.

# %% colab={"base_uri": "https://localhost:8080/", "height": 197} colab_type="code" id="8CXJd0ztTGRC" outputId="db9f4b2b-efdb-4b64-80e3-a24efa98049c"
corrupted_id_df = df[df["video_id"] == "#NAZWA?"]
for idx, t in enumerate(corrupted_id_df["publish_time"].unique()):
    corrupted_id_df.loc[corrupted_id_df["publish_time"] == t, "video_id"] = f"XXX{idx}"

df.loc[corrupted_id_df.index, :] = corrupted_id_df

df[df["video_id"].apply(lambda x: "XXX" in x)][["video_id", "title", "publish_time"]].head()

# %% [markdown] colab_type="text" id="gI1hhZuRZPUC"
# Now with the missing values fixed, we can look at UNIQUE values per column.
# <!--
# There are repeating entries for the same video, but only with different `trending_date`, as already shown.
#
# > We can aggregate those videos into one row, and replace `trending_date` column with:
#
# > - count of different trending_dates
# > - list of exact trending_dates  -->
# ### Can one `video_id` have more than one title?
# > Answer: YES ...
#
# > This shows that user who uploaded a video is able to change its title while it's listed in TRENDING.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 167} colab_type="code" id="Jn_vUSJrbxO7" outputId="b74dc3ff-6b21-4d83-df69-b8b18245e4a0"
df_by_video_id = df.groupby("video_id").agg({"title": lambda x: len(set(x))})
df_by_video_id.sort_values(by="title", ascending=False).head(3)

# %% colab={"base_uri": "https://localhost:8080/", "height": 167} colab_type="code" id="ltVrX64weytY" outputId="a1563f4c-80f1-403d-df0a-de3c79efd1a5"
print(df[df["video_id"] == "w_Ma8oQLmSM"].title.unique())

# %% [markdown] colab_type="text" id="gJuyyri1ESHj"
# ### Analyze distribution of 'category_id'

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} colab_type="code" id="D77HMXSWDtmi" outputId="0e507376-f3d1-4f83-9a3a-d0cad6a1ceeb"
from collections import Counter
import numpy as np

categories = df.category_id.values
nans = categories[np.isnan(categories)]
categories = categories[~np.isnan(categories)]
print("NANs:", nans.shape, "not NANs:", categories.shape)

df.hist(column="category_id", bins=int(max(categories)))
Counter(categories.tolist()).most_common()

# %% [markdown] colab_type="text" id="fUeS6cNTkZFz"
# ## Preview some categories examples
# - category **1** is trailers
# - category **2** is about cars and racing :P
# - category **10** is music videos
# - ...

# %% colab={"base_uri": "https://localhost:8080/", "height": 200} colab_type="code" id="dh1wGieakbDv" outputId="de910017-92ad-4a51-b085-4404eab5f540"
df[df["category_id"] == 24].head(10)["title"]

# %% [markdown] colab_type="text" id="ILzJC6_ayRj7"
# # TEXT Attributes
# - publish time
# - title
# - channel title
# - tags
# - description

# %% [markdown]
# ## 1. Publish Time
# A) Years of publish
#

# %%
import dateutil.parser

dates = [dateutil.parser.isoparse(d) for d in df["publish_time"].unique()]
years = [d.year for d in dates]
count_years = Counter(years)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_years.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])
ax.set_xticks([it[0] for it in its])
ax.set_title("Years of publish_time")

# %% [markdown]
# *B*) Month of publish
# > We can see that during some months, there have been much less trending videos than during other ones. In particular, months July to October (inclusive) are very rare.
#

# %%
import datetime 

months = [d.month for d in dates]
count_months = Counter(months)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_months.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([datetime.date(1900, it[0], 1).strftime('%B') for it in its], rotation=30)
ax.set_title("Months of publish_time")

# %% [markdown]
# C) Weekday
# > Conversely to what we expected, most trending videos have not been published on weekend

# %%
days = [d.weekday() for d in dates]
count_days = Counter(days)
weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

fig, ax = plt.subplots(figsize=(8, 6))
its =  count_days.items()
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([weekDays[it[0]] for it in its], rotation=30)
ax.set_title("WeekDays of publish_time")

# %%
# ADD week_day attribute
df["week_day"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).weekday())

# %% [markdown]
# D) Hour
# > There is a significant increase in trending videos in the middle of the day, between 13:00 and 19:00
#
# > Hours can be divided into four periods in a day:
# - 00:00 - 06:00 (time_of_day = 1)
# - 06:00 - 12:00 (time_of_day = 2)
# - 12:00 - 18:00 (time_of_day = 3)
# - 18:00 - 24:00 (time_of_day = 4)
#
# Let's make an attribute out of that.

# %%
hours = [d.hour for d in dates]
count_hours = Counter(hours)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_hours.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([f"{it[0]:02d}:00" for it in its], rotation=40)
ax.set_title("Hours of publish_time")


# %%
def extract_time_of_day(datestring):
  d = dateutil.parser.isoparse(datestring)
  return d.hour // 6 + 1

# ADD time_of_day attribute
df["time_of_day"] = df["publish_time"].apply(extract_time_of_day)
df[["time_of_day", "publish_time"]].head()

# ## 1. Publish Time
# A) Years of publish
#

# %%
import dateutil.parser

dates = [dateutil.parser.isoparse(d) for d in df["publish_time"].unique()]
years = [d.year for d in dates]
count_years = Counter(years)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_years.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])
ax.set_xticks([it[0] for it in its])
ax.set_title("Years of publish_time")

# %% [markdown]
# *B*) Month of publish
# > We can see that during some months, there have been much less trending videos than during other ones. In particular, months July to October (inclusive) are very rare.
#

# %%
import datetime 

months = [d.month for d in dates]
count_months = Counter(months)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_months.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([datetime.date(1900, it[0], 1).strftime('%B') for it in its], rotation=30)
ax.set_title("Months of publish_time")

# %%
# ADD month attribute
df["month"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).month)

# %% [markdown]
# C) Weekday
# > Conversely to what we expected, most trending videos have not been published on weekend

# %%
days = [d.weekday() for d in dates]
count_days = Counter(days)
weekDays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

fig, ax = plt.subplots(figsize=(8, 6))
its =  count_days.items()
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([weekDays[it[0]] for it in its], rotation=30)
ax.set_title("WeekDays of publish_time")

# %%
# ADD week_day attribute
df["week_day"] = df["publish_time"].apply(lambda x : dateutil.parser.isoparse(x).weekday())

# %% [markdown]
# D) Hour
# > There is a significant increase in trending videos in the middle of the day, between 13:00 and 19:00
#
# > Hours can be divided into four periods in a day:
# - 00:00 - 06:00 (time_of_day = 1)
# - 06:00 - 12:00 (time_of_day = 2)
# - 12:00 - 18:00 (time_of_day = 3)
# - 18:00 - 24:00 (time_of_day = 4)
#
# Let's make an attribute out of that.

# %%
hours = [d.hour for d in dates]
count_hours = Counter(hours)

fig, ax = plt.subplots(figsize=(8, 6))
its =  sorted(count_hours.items(), key=lambda x : -1*x[1])
rects = ax.bar([it[0] for it in its], [it[1] for it in its])

ax.set_xticks([it[0] for it in its])
ax.set_xticklabels([f"{it[0]:02d}:00" for it in its], rotation=40)
ax.set_title("Hours of publish_time")


# %%
def extract_time_of_day(datestring):
  d = dateutil.parser.isoparse(datestring)
  return d.hour // 6 + 1

# ADD time_of_day attribute
df["time_of_day"] = df["publish_time"].apply(extract_time_of_day)
df[["time_of_day", "publish_time"]].head()

# %% [markdown] colab_type="text" id="9di0iCsJyo6x"
# ## 2. Title
#
# **Lengths in characters**
# > On average, around 50 characters describe the title.
#
# > It seems as if there was normal distribution of title length in characters, with a right tail slightly longer.

# %% colab={"base_uri": "https://localhost:8080/", "height": 432} colab_type="code" id="vcIFgnRUQxnx" outputId="9d8ec612-0be9-4da7-810a-2dd576cedb2d"
import seaborn as sns

titles = df["title"].values
lengths = list(map(len, titles))

# ADD title_length_chars attribute
df["title_length_chars"] = df["title"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# %% colab={"base_uri": "https://localhost:8080/", "height": 50} colab_type="code" id="UhdbYmaBZKa9" outputId="6bc0013c-a1bc-46c3-ff55-719d6eb29c1c"
print("MAX length:", df.loc[df["title"].apply(len).idxmax(), :]["title"])
print()
print("MIN length:", df.loc[df["title"].apply(len).idxmin(), :]["title"])

# %% [markdown] colab_type="text" id="0SqoOgSmlvsA"
# **Lengths in words (*tokens*)**
# > There are 10 *words* in average describing a video's title.
#
# > There exist titles with only one word, and titles with a maximum of 28 words. 
#
# > The distirbution seems normal, but is not very smooth.

# %% colab={"base_uri": "https://localhost:8080/", "height": 432} colab_type="code" id="P36KDA5xyiFL" outputId="463fcdc8-3f75-4552-9c7e-a5d15481286e"
from nltk.tokenize import word_tokenize

lengths = []
for t in titles:
    lengths.append(len(word_tokenize(t)))

# ADD title_length_tokens attribute
df["title_length_tokens"] = df["title"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)
# %% [markdown]
# ### Upper vs. Lower case
# > We can observe a dominating ratio between uppercase letters and overall length of title: most of them have 20% uppercase characters. Right tail shows that higher ratios appear, but not as often as we could expect.


# %%
def get_uppercase_ratio(x):
    return sum([1 if char.isalpha() and char.isupper() else 0 for char in x]) / len(x)

# ADD title_uppercase_ratio attribute
df["title_uppercase_ratio"] = df["title"].apply(get_uppercase_ratio)

uppercase_ratio = df["title_uppercase_ratio"]
print(uppercase_ratio.describe())
sns.distplot(uppercase_ratio)


# %% [markdown]
# ### Non-alphanumeric characters
# > Similarily to uppercase ratio, there is a trend in titles that 20% of the characters describing it are neither letters nor digits, but other special characters.

# %%
def get_not_alnum_ratio(x):
    return sum([1 if not char.isalnum() else 0 for char in x]) / len(x)


# ADD title_uppercase_ratio attribute
df["title_not_alnum_ratio"] = df["title"].apply(get_not_alnum_ratio)

not_alnum_ratio = df["title_not_alnum_ratio"]
print(not_alnum_ratio.describe())
sns.distplot(not_alnum_ratio)

# %% [markdown]
# ### Top not-alphanumeric characters
# > For each often occurring character, we check in how many titles it was observed. Large percentages could suggest one should use this character in the title :)
#
# > We can create binary features which will tell for each title, whether it contains this particular character, or not. However, we don't want our feature vector to grow too much at this point, so instead of binary encoding for each common character, we will count all characters that belong to TOP-N characters. TOP-N will be derived by applying threshold: <u>more than 10% of titles</u>. Furthermore, we skip whitespace characters and assume that number of tokens will reflect this feature well enough.

# %%
count_chars = Counter("".join(titles))
print("Number of different characters:", len(count_chars.keys()))

top_chars = count_chars.most_common()
not_alnum = [t for t in top_chars if not t[0].isalnum()]
top_not_alnum = not_alnum[:15]

common_chars = []
for char, count in not_alnum:
    percentage = sum(df["title"].apply(lambda x : char in x)) / df.shape[0] * 100.0
    if percentage > 10.0:
        print(f"'{char}': {count}", ",", round(percentage, 3), "%")
        if not char.isspace():
            common_chars.append(char)

# %%
# ADD title_common_chars_count
df["title_common_chars_count"] = df["title"].apply(lambda x : sum(1 if char in common_chars else 0 for char in x))
sns.distplot(df["title_common_chars_count"])

# %% [markdown]
# ## 3. Channel title
# > For Channel title, we follow very similar analysis to **Title** analysis 
#
# > Channel title is usually less than 20 characters long

# %%
import seaborn as sns

titles = df["channel_title"].values
lengths = list(map(len, titles))

# ADD title_length_chars attribute
df["channel_title_length_chars"] = df["channel_title"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# %% [markdown]
# > Most Channel titles are one or two words long.

# %%
from nltk.tokenize import word_tokenize

lengths = []
for t in titles:
    lengths.append(len(word_tokenize(t)))

# ADD title_length_tokens attribute
df["channel_title_length_tokens"] = df["channel_title"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# %% [markdown]
# -

# %% [markdown]
# ## 4. Tags
#
# > For Tags, we simply apply **counting**. If **Tags** are `[none]`, we use $-1$ to denote this special value.
# > Looking at the distribution of tags counts, we can tell that there is no simple relation such as: the more tags the better.

# %%
df["tags_count"] = df["tags"].apply(lambda x : x.count('|') if '|' in x else -1)
sns.distplot(df["tags_count"])


# %% [markdown]
# #### Are there any popular tags?
# We apply `unique` in order to find popular tags,as if there was a video which was trending for many days, its tags are recorded multiple times. This would distort the distribution of tags across all the videos.
#
# > Some tags are more common than other ones, however most of them are connected to some fixed category, for example `trailer` - implies that the video content will be a movie trailer.

# %%
all_tags_list = df["tags"].apply(lambda x : [tag.lower().replace('"', '') for tag in x.split('|')]).values
print(len(all_tags_list))
all_tags_list = np.unique(all_tags_list)
print(len(all_tags_list))

all_tags = []
all_tags_tokens = []
for atl in all_tags_list:
    all_tags.extend(atl)
    for tag in atl:
        all_tags_tokens.extend(tag.split())
    
print("\nPOPULAR TAGS:")
print(Counter(all_tags).most_common(30))

# %%
print("POPULAR TAGS TOKENS:")
print(Counter(all_tags_tokens).most_common(30))


# %% [markdown]
# ## 5. Description
# **Lengths in characters**
#
# > Average length of description in characters is close to one thousand. However, the standard deviation is also very high, very close to the mean - roughly 845 characters. Median of the distribution is equal to 733 characters.
#
# > Almost no-descriptions also happen: minimum value observed is 1.

# %%
descriptions = df["description"].values
lengths = list(map(len, descriptions))

# ADD title_length_chars attribute
df["description_length_chars"] = df["description"].apply(len)

print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)

# %% [markdown]
# **Lengths in tokens**
# > On average, there are 156 words (tokens) per description. Maximum observation was 1520, which is a great outlier as the 75th percentile is equal to 212 words. 

# %%
lengths = []
for d in descriptions:
    lengths.append(len(word_tokenize(d)))

# ADD title_length_tokens attribute
df["description_length_tokens"] = df["description"].apply(lambda x : len(word_tokenize(x)))
    
print(pd.DataFrame({"length_statistics": lengths}).describe())
sns.distplot(lengths)


# %% [markdown]
# **Lengths in lines**
# > For description, we decided also to count newlines. This is because youtubers often format description in some way, for example enuemrating links to their social medias in new lines each.
#
# > It turned out that many descriptions have no new line at all. We suspect that sometimes there could have been an error when exporting the data.

# %%
def count_newlines(x):
    x = x.replace(r'\\n', r'\n')
    return x.count(r'\n')

# ADD title_length_tokens attribute
df["description_length_newlines"] = df["description"].apply(count_newlines)
    
print(df["description_length_newlines"].describe())
sns.distplot(df["description_length_newlines"])

# %% [markdown]
# **Upper vs. Lower case (again)**
# > We take a look onto uppercase letters again, followed by what we discovered in titles.
#
# > For the description attribute, the tendency towards using uppercase letters is smaller. Most ratios don't exceed 10%.

# %%
# ADD title_uppercase_ratio attribute
df["description_uppercase_ratio"] = df["description"].apply(get_uppercase_ratio)

uppercase_ratio = df["description_uppercase_ratio"]
print(uppercase_ratio.describe())
sns.distplot(uppercase_ratio)

# %% [markdown]
# **Number of URLs per description**
# > We observed that users often leave many links in their descriptions, such as: instagram profile, twitter profile, subscribe redirection url, or event download links. It may be useful to keep track on how many URLs each description has.
#
# > Although some descriptions lack URLs, the median count of URLs is equal to 7 - quite a lot - and the maximum observation is 83 URLs!

# %%
import re

best_url_regex = r'(http:\/\/www\.|http:\/\/|https:\/\/www\.|https:\/\/|www\.)?(?P<domain>([a-zA-Z0-9]+\.)+[a-zA-Z0-9]+)\/*[a-zA-Z0-9-_\/\.]*'
df["description_url_count"] = df["description"].apply(lambda x : len(re.findall(best_url_regex, x)))

print(df["description_url_count"].describe())
sns.distplot(df["description_url_count"])

# %% [markdown]
# #### Which URLs are popular?

# %%
descriptions = df["description"].unique()

domains = []
for d in descriptions:
    matches = re.findall(best_url_regex, d)
    for m in matches:
        if len(m) > 1:
            domains.append(m[1])

top_domains = Counter(domains).most_common(30)
print(top_domains)

# %% [markdown]
# #### For the most popular URLs, in how many unique descriptions (in %) do they appear?
# > Some URLs appear very often:
# - twitter.com
# - facebook.com
# - instagram.com
#
# We can hypothesise that trending videos include links to popular social medias. It is correlated with the fact that if some youtuber is popular, then she/he already has accounts on many other social medias as well, to reach out to more fans.

# %%
percentages = []
for domain, count in top_domains:
    percentage = sum(1 if domain in d else 0 for d in descriptions) / len(descriptions) * 100.0
    percentages.append((domain, percentage))

for domain, count in sorted(percentages, key=lambda x : -x[1]):
    print(f"{domain} : {count:.3f}%")

# %% [markdown] colab_type="text" id="cq93_rI8oVaA"
# ### Non-ascii characters
# # `TO DELETE?`

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} colab_type="code" id="eLmzFyXUmsOU" outputId="127c43f7-6a85-433b-aaff-d2fce0604ee9"
count_chars = Counter("".join(titles))
print("Number of different characters:", len(count_chars.keys()))
non_ascii = [key for key in count_chars if ord(key) > 127]
non_ascii_count = sorted([(key, count_chars[key]) for key in non_ascii], key=lambda x: -1 * x[1])
for pair in non_ascii_count[:15]:
    print(pair, ord(pair[0]))


# %% colab={"base_uri": "https://localhost:8080/", "height": 187} colab_type="code" id="hDzhW1cMqV5q" outputId="8235b5f5-e5f9-42a8-b5e3-262ea9186acc"
from sklearn.cluster import KMeans

codes = np.array(list(map(ord, [key for key in count_chars.keys() if ord(key) > 127]))).reshape(-1, 1)

nc = 3
kmeans = KMeans(n_clusters=nc)
kmeans.fit(codes)
y_kmeans = kmeans.predict(codes)

chars_clusters = list(zip(map(chr, codes.reshape(-1)), y_kmeans))
for i in range(nc):
    print("\nCLUSTER #", i)
    print(list(filter(lambda x: x[1] == i, chars_clusters))[:20])
    print(ord(list(filter(lambda x: x[1] == i, chars_clusters))[0][0]))


# %% [markdown]
# ### Emojis analysis

# %%
def read_emojis_txt(filename):
    emojis_df = pd.read_csv(filename, sep=";", header=None, comment="#")
    print(emojis_df[0].head())
    array = emojis_df[0].to_numpy()
    result = []
    for code in array:
        # range
        code = code.strip()
        if ".." in code:
            begin_code, end_code = code.split("..")
            for code in range(int(begin_code, 16), int(end_code, 16)):
                result.append(chr(code))
            pass
        elif not " " in code:
            result.append(chr(int(code, 16)))
    print(len(result))
    return set(result)
    

emojis = read_emojis_txt("emojis.txt")

# small test
"👻" in emojis


# %%
def count_emojis(text):
    return sum([1 for char in text if char in emojis])

df["emojis_counts"] = df["description"].apply(count_emojis)

# %%
print(df["emojis_counts"].describe())
sum(df["emojis_counts"] == 0), sum(df["emojis_counts"] != 0)

# %% [markdown]
# #### There is not much emojis in descriptions, anyway added column into consideration

# %%
sns.distplot(df["emojis_counts"][df["emojis_counts"] != 0])

# %% [markdown]
# ### Embeddings

# %%
import io
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

tf.__version__


# %%
#  encoder = info.features['text'].encoder
def write_embedding_files(labels, embedded_ndarray, path=path, prefix=""):
    out_v = io.open(os.path.join(path, f"{prefix}_vecs.tsv"), "w", encoding="utf-8")
    out_m = io.open(os.path.join(path, f"{prefix}_meta.tsv"), "w", encoding="utf-8")
    vectors = embedded_ndarray
    for message, vector in zip(labels, vectors):
        out_m.write(message + "\n")
        out_v.write("\t".join([str(x) for x in vector]) + "\n")
    out_v.close()
    out_m.close()


# %%
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# %%
messages = [
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding",
]

message_embeddings = embed(messages)

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    print("Message: {}".format(messages[i]))
    print("Embedding size: {}".format(len(message_embedding)))
    message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
    print("Embedding[{},...]\n".format(message_embedding_snippet))

# %%
unique_titles = np.unique(titles)
write_embedding_files(unique_titles, embed(unique_titles).numpy())

# %% [markdown]
# Embeding appropriate text columns

# %%
df.columns


# %%
def calc_embeddings(df, column_names, write_visualizations_files=False):
    for column in column_names:
        # batch_processing
        batch_size = 1000
        input_col = df[column].to_numpy()
        num_it = len(input_col) // batch_size

        result = np.zeros(shape=[len(input_col), 512])
        for i in range(num_it):
            result[batch_size * i: batch_size * (i + 1)]= embed(input_col[batch_size * i: batch_size * (i + 1)]).numpy()
        if len(input_col) % batch_size:
            result[batch_size * i:]= embed(input_col[batch_size * i:]).numpy()
        if write_visualizations_files:
            unique_inputs, unique_indexes = np.unique(input_col, return_index=True) 
            write_embedding_files(unique_inputs, result[unique_indexes], prefix=column)
        df[f"{column}_embed"] = list(result)

calc_embeddings(df, ["title", "channel_title"], False) # , "description" Description doesnt work...

# %%
np.unique([[0, 1], [0, 1]], axis=0)

# %%
pd.set_option("colwidth", 15)
print(df.head())
pd.set_option("colwidth", None)


# %% [markdown]
# ### Tags

# %%
def tags_transformer(x):
    return ", ".join(sorted([tag.replace('"', "") for tag in x.split("|")]))

df["transormed_tags"] = df["tags"].apply(tags_transformer)

# %%
df["transormed_tags"].head()

# %%
cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
print(df["transormed_tags"][0])
cosine_loss(embed([df["transormed_tags"][0]]), embed(["lewis, christmas, what, none, moz"]))

# %%
df.shape

# %%
calc_embeddings(df, ["transormed_tags"], False)

# %%
df.to_csv(os.path.join(path, "text_attributes_all_not_trending.csv"), index=False)
