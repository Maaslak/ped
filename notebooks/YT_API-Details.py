# %%

import json
import os
import pandas as pd

from googleapiclient.discovery import build

DATA_PATH = "data"

# %%

with open(os.path.join(DATA_PATH, "1224_related.json"), "r") as handle:
    loaded = json.load(handle)

# STORE YOUR YOUTUBE API KEY IN THE FILE BELOW!
with open(os.path.join('..', '..', 'tomato-soup.txt'), 'r') as file:
    api_key = file.read().replace("\n", "")

# api_key2 = ...
# api_key3 = ...

youtube = build('youtube', 'v3', developerKey=api_key)
# youtube2 = build('youtube', 'v3', developerKey=api_key2)
# youtube3 = build('youtube', 'v3', developerKey=api_key3)

# %%

response = youtube.videos().list(id='-6Zc8Co2H3w')

# %%

pd.read_csv(os.path.join(DATA_PATH))

# %%

GB_videos_df = pd.read_csv("data" + "/" + "GB_videos_5p.csv", sep=";", engine="python")
print(GB_videos_df.columns)
