# %%
import json
import os
import pandas as pd

DATA_PATH = "data"

# %%

with open(os.path.join(DATA_PATH, "detailed1.json"), "r") as handle:
    loaded = json.load(handle)

with open(os.path.join(DATA_PATH, "detailed2.json"), "r") as handle:
    loaded.extend(json.load(handle))

with open(os.path.join(DATA_PATH, "detailed3.json"), "r") as handle:
    loaded.extend(json.load(handle))

# %%

def parse_video(video):
    return [
        video["id"],  # video_id
        None,  # trending_date
        video["snippet"]["title"],  # title
        video["snippet"]["channelTitle"],  # channel_title
        video["snippet"]["categoryId"],  # category_id
        video["snippet"]["publishedAt"],  # publish_time
        "|".join([f'"{tag}"' for tag in video["snippet"]["tags"]]) if "tags" in video["snippet"] else "[none]",  # tags
        video["statistics"]["viewCount"] if "viewCount" in video["statistics"] else 0,  # views
        video["statistics"]["likeCount"] if "likeCount" in video["statistics"] else 0,  # likes
        video["statistics"]["dislikeCount"] if "dislikeCount" in video["statistics"] else 0,  # dislikes
        video["statistics"]["commentCount"] if "commentCount" in video["statistics"] else 0,  # comment_count
        video["snippet"]["thumbnails"]["default"]["url"],  # thumbnail_link
        False if 'commentCount' in video["statistics"] else True,  # comments_disabled
        False if 'likeCount' in video["statistics"] else True,  # ratings_disabled
        False,  # video_error_or_removed
        video["snippet"]["description"],
    ]


df_rows = []
for video_details in loaded:
    df_rows.append(parse_video(video_details))

# %%

df = pd.DataFrame(df_rows, columns= ['video_id', 'trending_date', 'title', 'channel_title', 'category_id',
       'publish_time', 'tags', 'views', 'likes', 'dislikes', 'comment_count',
       'thumbnail_link', 'comments_disabled', 'ratings_disabled',
       'video_error_or_removed', 'description '])

# %%

df.to_csv(os.path.join(DATA_PATH, "videos_not_trending.csv"))

# %%

new_df_test = pd.read_csv(os.path.join(DATA_PATH, "videos_not_trending.csv"))