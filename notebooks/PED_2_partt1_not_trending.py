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

# %%
import os

DOWNLOAD_IMAGES = False

path = "../data/"
files = os.listdir(path)
files


# %%
import pandas as pd


pd.set_option("colwidth", -1)

GB_videos_df = pd.read_csv(path + "/" + "GB_videos_5p.csv", sep=";", engine="python")
US_videos_df = pd.read_csv(path + "/" + "US_videos_5p.csv", sep=";", engine="python")

df = pd.concat([GB_videos_df, US_videos_df]).drop_duplicates().reset_index(drop=True)

# %%
df.columns

# %%
len(df['thumbnail_link'].unique()), len(df['thumbnail_link'])

# %%
df

# %% [markdown]
# ## Downloading images

# %%
images_path = os.path.join(path, "images")
try:
    os.mkdir(images_path)
except FileExistsError:
    pass

def url2filename(url):
    return url.replace("/", "")

# get content and write it to file
def write_to_file(filename, content):
    f = open(filename, 'wb')
    f.write(content)
    f.close()

from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

URLs = df['thumbnail_link'].unique()

import concurrent.futures
import urllib.request

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
def download_urls(urls, images_path, url2filename_func):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load_url, url, 60): url for url in URLs}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                if not "HTTP Error 404" in str(exc):
                    print('%r generated an exception: %s' % (url, exc))
            else:
                write_to_file(os.path.join(images_path, url2filename_func(url)), data)

if DOWNLOAD_IMAGES:
     download_urls(URLs, images_path, url2filename)
# %% [markdown]
# ## Images preview
# > There are 'missing values' in images as well - some thumbnails are default gray picture with a rectangle in the middle

# %%
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

sample_images = []

fig=plt.figure(figsize=(12, 24))
columns, rows = 4, 10
for i in range(1, columns*rows +1):
    image_path = os.path.join(images_path, os.listdir(images_path)[i*2])
    sample_images.append(image_path)
    
    img = mpimg.imread(image_path)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    
plt.show()

# %% [markdown]
# ### Display default picture

# %%
default_sample = mpimg.imread(os.path.join(images_path, "https_i.ytimg.comvi_0uV-jMbHQUdefault.jpg"))
fig = plt.figure(figsize=(3, 3))
plt.imshow(default_sample)
print(default_sample.shape)

# %% [markdown]
# ### Add boolean `has thumbnail` attribute; True or False if image is default thumbnail

# %%
df["image_filename"] = df["thumbnail_link"].apply(lambda x : x.replace('/', '').replace(':', '_'))

# %%
is_default_dict = {}
def check_if_image_is_default(thumbnail_link, default_sample):
    global is_default_dict
    images_path = '../data/images'
    image_filename = thumbnail_link.replace('/', '').replace(':', '_')
    if image_filename in is_default_dict.keys():
        return is_default_dict[image_filename]
    else:
        if os.path.isfile(os.path.join(images_path, image_filename)):
            img = mpimg.imread(os.path.join(images_path, image_filename))
            if img.shape != default_sample.shape:
                img = cv2.resize(img, default_sample.shape[:2][::-1], interpolation = cv2.INTER_AREA)
            diff_from_default = np.sum(np.power((img-default_sample) / 255.0, 2))
            is_default = diff_from_default < 100.0
            is_default_dict[image_filename] = is_default
            return is_default

df["has_thumbnail"] = df["thumbnail_link"].apply(lambda x : not check_if_image_is_default(x, default_sample))
default_thumbnails = df[df["has_thumbnail"] == False]["image_filename"].values
df.head(3)

# %% [markdown]
# ### (optional) - save list of default images to file, to avoid repeating the same work

# %%
default_thumbnails = df[df["has_thumbnail"] == False]["image_filename"].values

with open("../data/default_images.txt", "w") as handle:
    for dt in default_thumbnails:
        handle.write(dt)
        handle.write('\n')

# %% [markdown]
# ### Prepare `df_images` dataframe that will store only UNIQUE links to images, which are not default images

# %%
df_images = pd.DataFrame(
    {"image_filename": np.unique(df.loc[df["has_thumbnail"] == True, "image_filename"].values)}
)
df_images.head(5)
print(df_images.shape)

# %% [markdown]
# ### Load detection models
# > Tested: YOLO, tinyTOLO and Retinanet, however YOLO turned out to yield the best results.
#
# > Sometimes the results could be complementary to each other, but we skipped those detections as they were too costly computationally, while still uncertain if detections are correct AND useful.

# %%
from imageai.Detection import ObjectDetection

def get_yolo_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath('../models/yolo.h5')
    detector.loadModel()
    return detector

def get_retinanet_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath('../models/resnet50_coco_best_v2.0.1.h5')
    detector.loadModel()
    return detector

yolo_d = get_yolo_detector()
print("yolo loaded")
retinanet_d = get_retinanet_detector()
print("retinanet loaded")

detectors_dict = {
    "yolo" : yolo_d,
    "retinanet": retinanet_d,
}

import cv2

models_path = "../models"
face_cascade = cv2.CascadeClassifier(os.path.join(models_path, 'haarcascade_frontalface_default.xml'))
# %% [markdown]
# ### Test object detection on some sample images
# > Take only TOP 5 detections every time (sorted by probability)
#
# We decreased the `minimum_percentage_probability`, as the images are small and the models have some troubles detecting objects. In some images, all the probabilities would be below 50% (default threshold), while still accurate.

# %%
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# can use output_image_path to save results
for image_path in sample_images[:5]:
    img = mpimg.imread(image_path)
    # resized = resize_image(img, 200) # ? does not seem to work better ...
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    print()
    print(image_path)
    
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.show()
    
    for d_name in detectors_dict:
        print(d_name)
        detection = sorted(detectors_dict[d_name].detectObjectsFromImage(
                input_image=img,
                input_type="array",
                minimum_percentage_probability=25
            ), key=lambda x : -1*x["percentage_probability"]
        )

        for eachItem in detection[:5]:
            print("\t", eachItem["name"] , " : ", eachItem["percentage_probability"])
            
    detections = face_cascade.detectMultiScale(gray, 1.03, 3)
    print(f"Haar Cascade faces -> number of detections = {len(detections)}")


# %% [markdown]
# ### Perform YOLO detection on all thumbnails
# > `WARNING`: This is taking a lot of time.
# #### Result of running the cell below is written to `yolo_detections.json` file for convenience

# %%
image_filenames = df_images["image_filename"].values
yolo_detections = []

# for idx, i in enumerate(image_filenames):
for idx, i in enumerate(image_filenames[:5]):
    image_path = os.path.join(images_path, i)
    img = mpimg.imread(image_path)
    detection = sorted(yolo_d.detectObjectsFromImage(
            input_image=img,
            input_type="array",
            minimum_percentage_probability=25
        ), key=lambda x : -1*x["percentage_probability"]
    )[:5] # assumption - take top 5 most confident
    classes = [item["name"] for item in detection]
    yolo_detections.append(classes)
    print(f'"{i}": {str(classes)},')

# %% [markdown]
# ### Load `json` storing YOLO detections
# > What are the most common detections?

# %%
import json
with open("..\data\yolo_detections.json", "r") as handle:
    yolo_detections_dict = json.load(handle)
    
all_detections = []
for key in yolo_detections_dict:
    all_detections.extend(yolo_detections_dict[key])

from collections import Counter
Counter(all_detections).most_common(15)

# %% [markdown]
# What are all possible classes that were found?

# %%
print(set(all_detections), "\n", "number of classes:", len(set(all_detections)))

# %% [markdown]
# #### Which detections are actually useful?
# - Only one class is dominant: 'person'
# - Another one, **TV**, might also be useful, but we should keep in mind that TV seems hard to detect
# - For the remaining ones, we can define custom **groups** of classes, or assign "other" group to the rest
#
# Possible features:
# - if there was any detection at all
# - if there was a detection of person
# - if there was a detection of something else than person
#
# Possible groups:
# - vehicles
# - animals
# - food 

# %%
vehicles = "bicycle,   car,   motorcycle,   airplane, bus,   train,   truck,   boat "
vehicles = list(map(lambda x : x.strip(), vehicles.split(',')))

animals = " bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra, giraffe  "
animals = list(map(lambda x : x.strip(), animals.split(',')))

food = "  banana,   apple,   sandwich,   orange, broccoli,   carrot,   hot dog,   pizza,   donut,   cake  "
food = list(map(lambda x : x.strip(), food.split(',')))

# %%
df_images["has_detection"] = df_images["image_filename"].apply(lambda x : len(yolo_detections_dict[x]) > 0)
df_images["person_detected"] = df_images["image_filename"].apply(lambda x : "person" in yolo_detections_dict[x])
df_images["object_detected"] = df_images["image_filename"].apply(lambda x : len(yolo_detections_dict[x]) > 0 and any(d != "person" for d in yolo_detections_dict[x]))

df_images["vehicle_detected"] = df_images["image_filename"].apply(lambda x : any(d in vehicles for d in yolo_detections_dict[x]))
df_images["animal_detected"] = df_images["image_filename"].apply(lambda x : any(d in animals for d in yolo_detections_dict[x]))
df_images["food_detected"] = df_images["image_filename"].apply(lambda x : any(d in food for d in yolo_detections_dict[x]))

df_images.head(4)

# %% [markdown]
# #### How many rows have `True` values in the newly created attributes?
# - any_detection: 81% 
# - person: 71%
# - vehicles: almost 4% ...

# %%
for cname in df_images.columns:
    if 'detect' in cname:
        print(cname, " : ", df_images[cname].sum(), " : ", f"{df_images[cname].sum() / df_images.shape[0] * 100.0:.2f}%")

# %% [markdown]
# ### Perform Haar Cascade face detection on all thumbnails

# %%
image_filenames = df_images["image_filename"].values
face_detections = []

for idx, i in enumerate(image_filenames):
    image_path = os.path.join(images_path, i)
    img = mpimg.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    detections = face_cascade.detectMultiScale(gray, 1.03, 3)
    face_detections.append(len(detections))

# %%
face_detections = np.asarray(face_detections)
n, bins, patches = plt.hist(face_detections, density=True, facecolor='b', alpha=0.75)

plt.grid(True)
plt.show()

# %% [markdown]
# The histogram shows that usually no face is detected, however, the algorithm manages to detect a face in some cases. Therefore, we will use it as an additional feature - single number.

# %% [markdown]
# #### Add attribute: `face_count` based upon Haar Cascade algorithm

# %%
fd_dict = dict(zip(image_filenames, face_detections))
df_images["face_count"] = df_images["image_filename"].apply(lambda x : fd_dict[x])
df_images.head(10)

# %% [markdown]
# ## Images metadata - median pixel values and histograms (`5 bins`)
#
# We can analyze images on lower level than objects/text detection. 
#
# Pixel values can tell about the dominant color, or level of brightness or saturation.
#
# - Basic approach is to take median pixel values for all the images. We can characterize the images overall by analyzing the distribution of this median.
#
#
# - More in-depth approach would create a histogram of pixel values for each image. We don't need to use many bins to be able to tell if an image is rather bright or dark. Values of those 5 bins can be a feature of an image.

# %% [markdown]
# ### A) When converted to grayscale

# %%
img_cache = {}
def get_gray_histogram(image_filename):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[11:79, :] # remove top and bottom black rectangles
        img_cache[image_filename] = gray
    else:
        gray = img_cache[image_filename]
    hist, _ = np.histogram(gray.flatten(), bins=5)
    return hist / len(gray.flatten())

def get_median_gray_value(image_filename):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[11:79, :] # remove top and bottom black rectangles
        img_cache[image_filename] = gray
    else:
        gray = img_cache[image_filename]
    return np.median(gray.flatten())

df_images.loc[:, "gray_histogram"] = df_images["image_filename"].apply(get_gray_histogram)
df_images.loc[:, "gray_median"] = df_images["image_filename"].apply(get_median_gray_value)

df_images.head(3)

# %% [markdown]
# #### Median pixel value of an image
#
# Looking at the histogram, we can tell that most images median value is something in the middle of pixel range: 0.0 - 255.0. However, there is a tendency towards darker images - left tail of the histogram is thicker.

# %%
gray_medians = df_images.loc[:, "gray_median"].values
BINS = 10
n, bins, patches = plt.hist(gray_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor((i/10.0, i/10.0, i/10.0))

# %% [markdown]
# #### More details - plot distributions of each histogram bin separately
# How is the visualization below constructed:
# - for each image, a histogram is computed with 5 bins
#
# - after all histograms are computed, we can separately aggreagate values from first, second ... fifth bin from those histograms, and display those distributions on separate plots (also histograms :))
#
# The histograms somehow reflect the more general plot above. We can see the darker shades domination.
#
# There are not many bright images, while darker images exist. Looking at the lowest pixel range, 0.0-51.0 (where max=255.0) - leftmost plot, a bin with the highest value corresponds to 20%-40%; second highest - 40%-60%. Looking at the next histograms, the leftmost bin, representing 0%-20% range, is getting higher and higher value.

# %%
gray_histograms = np.stack(df_images.loc[:, "gray_histogram"].values)

fig, axes = plt.subplots(1, 5, figsize=(24, 4))

for i in range(len(axes)):
    axes[i].set_ylim(0, 7500)
    BINS = [i/5 for i in range(5)]
    n, bins, patches = axes[i].hist(gray_histograms[:, i], edgecolor='black', color=str(i / 5), bins=BINS)
    axes[i].set_title(f"pixel values {i/5*255.0} - {(i+1)/5*255.0}")
    axes[i].set_xlim(0.0, 1.0)
    axes[i].set_xticks(BINS)

plt.show()

# %% [markdown]
# ### B) When converted to HSV
#
# We can perform similar analysis in HSV color scale, as it is way more interpretable than RGB.
#
# HSV representation provides three interesting channels:
# - Hue
# - Saturation
# - Value

# %%
img_cache = {}
def get_hsv_histogram(image_filename, channel="hue"):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv[11:79, :, :] 
        img_cache[image_filename] = hsv
    else:
        hsv = img_cache[image_filename]
    c = 0 if channel == "hue" else 1 if channel == "saturation" else 2 if channel == "value" else -1
    hist, _ = np.histogram(hsv[:, :, c].flatten(), bins=5)
    return hist / len(hsv[:, :, c].flatten())

def get_median_hsv_value(image_filename, channel="hue"):
    global img_cache
    if image_filename not in img_cache:
        img = mpimg.imread(os.path.join(images_path, image_filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = hsv[11:79, :, :] 
        img_cache[image_filename] = hsv
    else:
        hsv = img_cache[image_filename]
    c = 0 if channel == "hue" else 1 if channel == "saturation" else 2 if channel == "value" else -1
    return np.median(hsv[:, :, c].flatten())

df_images.loc[:, "hue_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="hue"))
df_images.loc[:, "hue_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="hue"))

df_images.loc[:, "saturation_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="saturation"))
df_images.loc[:, "saturation_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="saturation"))

df_images.loc[:, "value_histogram"] = df_images["image_filename"].apply(lambda x : get_hsv_histogram(x, channel="value"))
df_images.loc[:, "value_median"] = df_images["image_filename"].apply(lambda x : get_median_hsv_value(x, channel="value"))

df_images.head()

# %% [markdown]
# #### B) a) HUE
#
# We can distinguish two dominant colors looking at the distribution of median HUE value across all the images.
#
# 1. RED shades
# 2. BLUE shades
#
# Green and pink medians are rare.

# %%
import matplotlib.colors

hue_medians = df_images.loc[:, "hue_median"].values
BINS = 10
n, bins, patches = plt.hist(hue_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((i/10, 1.0, 1.0)))

# %% [markdown]
# #### More details - plot distributions of each histogram bin separately
#
# Looking at the detailed histograms, we can observe that:
# - some images dominant colors are shades of RED, ORANGE, and some YELLOW-ORANGE (even up to 80%-100% of the pixels)
# - most images have very little YELLOW-GREEN shades
# - most images have mild addition of GREEN, BLUE and VIOLET; their distribution is very similar (usually 20% of all colors distribution)
#
# <u>On the plot below, the colors do not refer to the bins at all - the bins are colored to represent the color range</u>

# %%
hue_histograms = np.stack(df_images.loc[:, "hue_histogram"].values)

fig, axes = plt.subplots(1, 5, figsize=(24, 4))

import matplotlib.colors
from matplotlib.pyplot import cm


for i in range(len(axes)):
    axes[i].set_ylim(0, 8000)
    n, bins, patches = axes[i].hist(
        hue_histograms[:, i], 
        edgecolor='black', 
        color=matplotlib.colors.hsv_to_rgb((i/5, 1.0, 1.0)), 
        bins=5,
    )

    for p in range(len(patches)):
        patches[p].set_facecolor(matplotlib.colors.hsv_to_rgb(((i+(p/len(patches)))/5, 1.0, 1.0)))
    
plt.show()

# %% [markdown]
# #### B) b) SATURATION

# %% [markdown]
# Analysis of median SATURATION of all images show that most images are not highly saturated. Highest values cummulate below the middle value of saturation available. A histogram has a right tail longer, with reflects the tendency towards mild saturation.

# %%
saturation_medians = df_images.loc[:, "saturation_median"].values
BINS = 10
n, bins, patches = plt.hist(saturation_medians, edgecolor='black', bins=10)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((1.0, i/10.0, 1.0)))

# %% [markdown]
# #### B) c) VALUE
#
# Median of the VALUE seems to be normally distributed. Medium values are the most frequent, and the edge values are the least.

# %%
value_medians = df_images.loc[:, "value_median"].values
BINS = 15
n, bins, patches = plt.hist(value_medians, edgecolor='black', bins=BINS)

for i in range(len(patches)):
    patches[i].set_facecolor(matplotlib.colors.hsv_to_rgb((0.75, 0.75, i/BINS)))


# %% [markdown]
# ### Edge detection 
# We can try to describe DYNAMICS of an image by applying Canny filter.
#
# We add `edges` attribute which is a ratio of edges detected to all pixels in an image. Lower values mean that the image is smooth. Higher values suggest that the image is dynamic and the frequencies change often.

# %%
def get_edges(image_filename):
    img = mpimg.imread(os.path.join(images_path, image_filename))
    img = img[11:79, :, :]
    edges = cv2.Canny(img, 100, 200) / 255.0
    return np.sum(edges) / edges.size

df_images.loc[:, "edges"] = df_images["image_filename"].apply(get_edges)
df_images.head(3)


# %% [markdown]
# It turns out that the distribution of such ratio is pretty normal. Usually the edges ratio is around 20-25%.

# %%
edges = df_images.loc[:, "edges"].values
BINS = 15
n, bins, patches = plt.hist(edges, edgecolor='black', bins=BINS)

# %% [markdown]
# ## Save DF to file

# %%
df_images.head()

# %%
df_images.to_csv("..\data\image_attributes.csv", index=False)

# %% [markdown]
# ### Voila!
