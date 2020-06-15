# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: ped-venv
#     language: python
#     name: ped-venv
# ---

# +
import os
from apiclient.discovery import build

# STORE YOUR YOUTUBE API KEY IN THE FILE BELOW!
with open(os.path.join('..', '..', 'tomato-soup.txt'), 'r') as file:
    api_key = file.read()
    
api_key2 = ...
api_key3 = ...
    
youtube = build('youtube', 'v3', developerKey = api_key)
youtube2 = build('youtube', 'v3', developerKey = api_key2)
youtube3 = build('youtube', 'v3', developerKey = api_key3)

import json

with open("API_categories.json", "r") as handle:
    ids_to_categories_dict = json.load(handle)
   
with open("US_cat10.json", "r") as handle:
    x = json.load(handle)
    print("Videos for category 10, US:", len(x))
    
with open("US_cat24.json", "r") as handle:
    x = json.load(handle)
    print("Videos for category 24, US:", len(x))
    
with open("1213_related.json", "r") as handle:
    loaded = json.load(handle)
    used_ids = [l['relatedTo'] for l in loaded]    
    
with open("1209_related.json", "r") as handle:
    loaded.extend(json.load(handle))
    used_ids = [l['relatedTo'] for l in loaded]    
    
with open("1202_related.json", "r") as handle:
    loaded.extend(json.load(handle))
    used_ids = [l['relatedTo'] for l in loaded]   
    
with open("1224_related.json", "r") as handle:
    loaded.extend(json.load(handle))
    used_ids = [l['relatedTo'] for l in loaded]    
    
with open("1210_related.json", "r") as handle:
    loaded.extend(json.load(handle))
    used_ids = [l['relatedTo'] for l in loaded]    
    print("Loaded by related:", len(loaded))
    print("Ids used:", len(set(used_ids)))

# +
import pandas as pd

path = "../data/"
GB_id_titles = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")#.loc[:, ["video_id", "title", "category_id", "publish_time"]]
US_id_titles = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")#.loc[:, ["video_id", "title", "category_id", ]]

GB_id_titles["category_id_API"] = GB_id_titles["video_id"].apply(lambda x : ids_to_categories_dict.get(x, -1))
GB_id_titles["publish_time"] = GB_id_titles["publish_time"].apply(pd.to_datetime)
GB_id_titles["region_code"] = "GB"

US_id_titles["category_id_API"] = US_id_titles["video_id"].apply(lambda x : ids_to_categories_dict.get(x, -1))
US_id_titles["publish_time"] = US_id_titles["publish_time"].apply(pd.to_datetime)
US_id_titles["region_code"] = "US"

GB_year_filtered = GB_id_titles[GB_id_titles["publish_time"].apply(lambda x : x.year >= 2017 and x.year <= 2018)]
US_year_filtered = US_id_titles[US_id_titles["publish_time"].apply(lambda x : x.year >= 2017 and x.year <= 2018)]

id_titles_df = pd.concat([GB_year_filtered, US_year_filtered]).drop_duplicates().reset_index(drop=True)

df = pd.read_csv("aggregated.csv")
video_ids = df["video_id"].values
video_ids[:5]
# -

GB_year_filtered.video_id.unique().shape, US_year_filtered.video_id.unique().shape

id_titles_df["publish_time"].min(), id_titles_df["publish_time"].max()

# ### Select `publish_time` based on analysis in `PED.ipynb`:
# - take some from **2017** and some from **2018** (1 : 2 ratio) - skip OLDER
# - months: as the histogram tells, the least for June-October, most videos from Nov, Dec, Jan. I think the distributions should match

# +
agg_statistics_df = id_titles_df \
    .loc[:, ["video_id", "views", "likes", "dislikes", "comment_count", "comments_disabled", "ratings_disabled", "category_id_API", "region_code"]] \
    .groupby(by="video_id") \
    .agg(
        views=("views", "max"), 
        likes=("likes", "max"), # how many days this video was trending,
        dislikes=("dislikes", "max"), # how many days this video was trending,
        comment_count=("comment_count", "max"), # how many days this video was trending,
        comments_disabled=("comments_disabled", "mean"), # how many days this video was trending,
        ratings_disabled=("ratings_disabled", "mean"), # how many days this video was trending,
        category_id_API=("category_id_API", "max"),
        region_code=("region_code", "max")
    ).reset_index()

agg_statistics_df = agg_statistics_df[agg_statistics_df["video_id"] != "#NAZWA?"]
agg_statistics_df = agg_statistics_df[agg_statistics_df["video_id"] != "#NAZWA?"]

agg_statistics_df

# +
import seaborn as sns

US_agg_df = agg_statistics_df[agg_statistics_df["region_code"] == "US"]
sns.countplot(US_agg_df["category_id_API"])
# -

GB_agg_df = agg_statistics_df[agg_statistics_df["region_code"] == "GB"]
sns.countplot(GB_agg_df["category_id_API"])

# +
import random
from googleapiclient.errors import HttpError

# videos_by_category = {category : [] for category in US_agg_df["category_id_API"].value_counts().index.tolist() if category != -1}
# relatedToVideoId
sample_list = agg_statistics_df.loc[:, ["video_id", "category_id_API", "region_code"]].values.tolist()
quota_exceeded = []

yt_available = [youtube, youtube2, youtube3]
yt_COUNT = 3
yt = yt_available.pop()

results = []
corresponding_samples = []

while len(quota_exceeded) != yt_COUNT:
    video_id, category, region_code = random.choice(sample_list)
    if video_id in used_ids:
        continue
    
    if category != -1:
        try:
            request = yt.search().list(
                part="snippet",
                publishedAfter="2017-01-15T00:00:00Z",
                publishedBefore="2018-05-01T00:00:00Z",
                regionCode=region_code,
                safeSearch="none",
                type="video",
                videoCategoryId=str(category),
                relatedToVideoId=video_id
            )
            response = request.execute()
            if 'items' in response.keys():
                print(video_id, "success!")
                results.append(response['items'])
                corresponding_samples.append(video_id)
        except HttpError as err:
            print(str(err))
            if err.resp.status == 403:
                quota_exceeded.append(yt)
                if yt_available:
                    yt = yt_available.pop()
            
        

# +
flat_results = []
for r, s in zip(results, corresponding_samples):
    for item in r:
        item["relatedTo"] = s
        flat_results.append(item)
        
len(flat_results)
# -

with open(f"1224_related.json", "w") as handle:
    json.dump(flat_results, handle)
