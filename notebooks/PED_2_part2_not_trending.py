# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% id="2BnDJ63NW_dz" colab_type="code" outputId="66794567-8d08-471f-bdb7-49ae0e0026ac" colab={"base_uri": "https://localhost:8080/", "height": 54}
from google.colab import drive
drive.mount('/content/drive')

# %% id="k16eqDS5aB5_" colab_type="code" outputId="612a2bf9-c474-42a0-8621-fd41b3c39196" colab={"base_uri": "https://localhost:8080/", "height": 34}
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# %% id="z0WIQLIcW-Ww" colab_type="code" outputId="b89ffe55-af86-4c82-cb38-5528f3806ad7" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# !pip install --user keras_ocr
# !pip install --user pyspellchecker
# !pip install --user face_recognition
# !ls -l '/content/drive/My Drive/data/PED/data'

# %% id="_siiVEelW1df" colab_type="code" outputId="a3f30df6-7055-4c3d-fb81-260ddff01ef5" colab={"base_uri": "https://localhost:8080/", "height": 544}
import os

DOWNLOAD_IMAGES = False
GENERATE_OCR = False

BASE_PATH = '/content/drive/My Drive/data/PED/'
path = os.path.join(BASE_PATH, 'data')
models_path = os.path.join(BASE_PATH, "models")
files = os.listdir(path)
files


# %% id="OT56UuuFW1dn" colab_type="code" colab={}
import pandas as pd

pd.set_option("colwidth", None)

df = pd.read_csv(path + "/" + "videos_not_trending.csv")

# %% id="nHNlu095W1dr" colab_type="code" outputId="db8bcc32-4991-4deb-95b5-4dec77f5b52f" colab={"base_uri": "https://localhost:8080/", "height": 102}
df.columns

# %% id="lqPOpg6tW1dv" colab_type="code" outputId="50178de9-6bf0-473f-d0d0-a703b3f6aa7b" colab={"base_uri": "https://localhost:8080/", "height": 34}
len(df['thumbnail_link'].unique()), len(df['thumbnail_link'])

# %% id="qC_vFmLvW1dz" colab_type="code" outputId="357548a0-03ee-40c2-8ece-04190f780753" colab={"base_uri": "https://localhost:8080/", "height": 119}
df['video_id'].head()

# %% [markdown] id="iIDerrnNW1d2" colab_type="text"
# # Thumbnails analysis

# %% [markdown] id="lY-ZO7hDW1d3" colab_type="text"
# ## Downloading images

# %% id="D9UBPeC1W1d4" colab_type="code" colab={}
images_path = os.path.join(path, "images2")
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
# %% [markdown] id="dZ6hd6SpW1d7" colab_type="text"
# ## Text recognition


# %% id="N6n9j8jaW1d9" colab_type="code" colab={}
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_ocr
import pathlib

def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
# run the line below if you're using local runtime and have GTX > 1660 (this is known bug with tensorflow memory allocation)
# allow_memory_growth()


# %% id="3ruE1gMjW1eA" colab_type="code" outputId="1e6ef3e1-4651-4cc6-ec30-4162639a04e1" colab={"base_uri": "https://localhost:8080/", "height": 156}
pipeline = keras_ocr.pipeline.Pipeline()


# %% id="YSlK9xJKW1eE" colab_type="code" outputId="cee88dc3-04fb-43d1-cd1a-c975d43d4887" colab={"base_uri": "https://localhost:8080/", "height": 34}
data_dir = pathlib.Path(images_path)
print(data_dir)

# %% id="ML4vk8LOW1eH" colab_type="code" outputId="582414d8-dbe3-4b61-d3e5-549ecde13483" colab={"base_uri": "https://localhost:8080/", "height": 34}
image_count = len(list(data_dir.glob('*.jpg')))
image_count

# %% id="aqNnS-zIW1eL" colab_type="code" colab={}
list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))


# %% id="jeFX-B-jW1eP" colab_type="code" colab={}
AUTOTUNE = 10
BATCH_SIZE = 100
IMG_WIDTH, IMG_HEIGHT = 90, 120

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, file_path

ds = list_ds.map(process_path) # , num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE)
ds = ds.repeat()

# %% id="efKjZydAW1eS" colab_type="code" colab={}
import json
ocr_file_path = os.path.join(path, 'filename2text_not_trending.json')
def genarete_texts_dict(ds, num_steps):
    descriptions = {}
    for i, (img_batch, filenames_batch) in enumerate(ds):
        print(f"Step {i + 1}/{num_steps}")
        if i == num_steps:
            break
        prediction_groups = pipeline.recognize(img_batch.numpy() * 256)
        texts = [" ".join([item[0] for item in group])for group in prediction_groups] # joining words into sentence
        for filename, text in zip(filenames_batch.numpy(), texts):
            descriptions[filename.decode().split("/")[-1][:-4]] = text
    return descriptions

if GENERATE_OCR:
    filename2text = genarete_texts_dict(ds, int(image_count/BATCH_SIZE) + 1)
    with open(ocr_file_path, 'w') as fp:
        json.dump(filename2text, fp)
else:
    with open(ocr_file_path, 'r') as f:
        filename2text = json.load(f)

# %% [markdown] id="L4VFwkmxW1eV" colab_type="text"
# ### Correcting words
#
# #### Create known words from descriptions

# %% id="MNfOXbxIW1eW" colab_type="code" colab={}
descriptions_path = os.path.join(path, 'descriptions.csv')
df['description '].to_csv(descriptions_path, index=False, sep=" ")

# %% id="uWfssP3LW1eZ" colab_type="code" colab={}
import re 
preprocessed_descriptions = pd.Series([
    " ".join([word for word in re.split(r"[^a-zA-Z]", sentence.replace("\\n", " ")) if len(word) < 15 and len(word) > 2])
    for sentence in df['description '].unique() if type(sentence) == str
])
preprocessed_descriptions.to_csv(descriptions_path, index=False, sep=" ")

# %% id="OAdsvUpnW1ec" colab_type="code" colab={}
from spellchecker import SpellChecker

spell = SpellChecker()
spell.word_frequency.load_text_file(descriptions_path)

# %% id="Y-BAZ0-HW1ef" colab_type="code" outputId="35a1a745-24a6-4556-be05-bf725f50ba8f" colab={"base_uri": "https://localhost:8080/", "height": 34}
spell.word_frequency._longest_word_length

# %% id="dN5lvyDkW1ej" colab_type="code" outputId="46fe6ed7-73de-431b-a7a1-eb09f5a5ff5e" colab={"base_uri": "https://localhost:8080/", "height": 493}
import timeit
# filename2text_not_trending.json
thumbnails_path = os.path.join(path, 'corrected_thumbnail_not_trending.json')
start = timeit.default_timer()
if GENERATE_OCR:
    result = {}
    for i, (key, value) in enumerate(filename2text.items()):
        if i % 500 == 0:
            print(f"{i}/{len(filename2text)}")
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            start = stop
        if value:
            result[key] = " ".join([spell.correction(word) for word in value.split(" ")])
        else :
            result[key] = value
    with open(thumbnails_path, 'w') as fp:
        json.dump(result, fp)
else:
    with open(thumbnails_path) as fp:
        result = json.load(fp)

# %% id="PssdwQL0W1em" colab_type="code" outputId="468508fb-ee34-463a-fdab-0b804782722a" colab={"base_uri": "https://localhost:8080/", "height": 119}
list(result.keys())[:3], list(filename2text.keys())[:3]


# %% [markdown] id="yFQC6L_UW1ep" colab_type="text"
# #### Visualization of results

# %% id="QqSCGNN-W1ep" colab_type="code" outputId="d521ddca-105d-4665-de14-ffac1de19d0b" colab={"base_uri": "https://localhost:8080/", "height": 670}
def show_batch(image_batch, labels=None):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.axis('off')
        if labels:
            plt.title(labels[n])
    plt.tight_layout()


image_batch, filenames = next(iter(ds))

show_batch(image_batch.numpy(), [result[str(filename).split("/")[-1][:-5]] for filename in filenames[:25].numpy()])


# %% id="YqdeAd6VW1es" colab_type="code" colab={}
def url2filename_hq(url):
    url = url.replace("default.jpg", "hqdefault.jpg")
    return result[url2filename(url)[:-4]] if url2filename(url)[:-4] in result else ""

df['thumbnail_ocr'] = df['thumbnail_link'].map(url2filename_hq)

# %% id="BPywRBK3W1ev" colab_type="code" outputId="d05c1b89-3ff6-4c21-de94-fc262f452209" colab={"base_uri": "https://localhost:8080/", "height": 170}
df['thumbnail_ocr'].describe(), list(result.keys())[:4]

# %% [markdown] id="lTHY_0BVW1ey" colab_type="text"
# #### Apply text processing

# %% id="UOdSsVloW1ez" colab_type="code" outputId="c0bbcbed-f934-4b9c-f6aa-e75352a73a69" colab={"base_uri": "https://localhost:8080/", "height": 34}
import io
import os

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

tf.__version__


# %% id="oiO3OXGTW1e2" colab_type="code" colab={}
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


# %% id="eLe1Kqv1W1e7" colab_type="code" colab={}
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


# %% id="RgLZaMvvW1e_" colab_type="code" colab={}
def calc_embeddings(df, column_names, write_visualizations_files=False):
    extended_df = df
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
        extended_df[f"{column}_embed"] = list(result)
    return extended_df

extended_df = calc_embeddings(df, ["thumbnail_ocr"], True) # , "description" Description doesnt work...

# %% [markdown] id="URHKUmlOW1fC" colab_type="text"
# #### OCR lengths
#
# > Conclusion: There are a lot of thumbnails without text detected, but median and mean values show that there are aproximately 1 word per image,
#     which could be informative as well

# %% id="25QT3xVmW1fC" colab_type="code" outputId="23c06b47-d455-49d3-abe6-396489fd7f49" colab={"base_uri": "https://localhost:8080/", "height": 527}
import seaborn as sns
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

lengths = []
for t in df["thumbnail_ocr"].to_numpy():
    lengths.append(len(word_tokenize(t)))

print(pd.DataFrame({"length_statistics": lengths}).describe())
df["thumbnail_ocr_length"] = df["thumbnail_ocr"].apply(lambda x : len(word_tokenize(x)))
sns.distplot(lengths)

# %% [markdown] id="Ogx8yvnPW1fG" colab_type="text"
# # Emotions analysis

# %% id="7L04ggQAW1fK" colab_type="code" colab={}
import face_recognition

# %% id="COX1mmZyW1fN" colab_type="code" outputId="9259a07f-4d0c-4069-f9d3-f5ea340d0e69" colab={"base_uri": "https://localhost:8080/", "height": 361}
image_batch.numpy()[0].shape
plt.hist(np.reshape(image_batch.numpy(), [-1]))

# %% [markdown] id="C-XWenHSW1fR" colab_type="text"
# #### First face at the image visualization

# %% id="eI5LAWSSW1fS" colab_type="code" outputId="0451f26a-b2fa-40dc-d6ce-8d0b82ef9c9d" colab={"base_uri": "https://localhost:8080/", "height": 268}
image_batch, filenames = next(iter(ds))
plt.imshow(image_batch[0])
plt.show()

# %% id="dOGg01hZW1fV" colab_type="code" outputId="221538a8-0753-4681-f798-9dc3b51386f0" colab={"base_uri": "https://localhost:8080/", "height": 265}
image = tf.image.convert_image_dtype(image_batch, tf.uint8).numpy()[0]
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model="cnn")
if face_locations:
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    plt.imshow(face_image)
    plt.show()

# %% id="ESpke5gEW1fc" colab_type="code" colab={}
from tensorflow.keras.models import load_model

# %% id="KG1CHLrIW1ff" colab_type="code" outputId="94dd90d9-d8c6-4ffc-a8a7-fa3b6a11d594" colab={"base_uri": "https://localhost:8080/", "height": 34}
face_locations

# %% id="pRnPxr6CW1fi" colab_type="code" colab={}
face_recognition_model = load_model(os.path.join(models_path, "face_recognition.hdf5"))

# %% [markdown] id="3V6BzibTW1fl" colab_type="text"
# #### Detected emotion

# %% id="ePLGswZIW1fm" colab_type="code" outputId="3d5b0952-3b30-4e9d-ecc9-81fb9fc108a6" colab={"base_uri": "https://localhost:8080/", "height": 34}
emotions_map = {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
emotions_map_inv = {value: key for key, value in emotions_map.items()}
emotions_map_inv[np.argmax(face_recognition_model.predict(tf.image.rgb_to_grayscale(tf.image.resize([face_image], [48, 48]))))]

# %% id="Xxn1GxmQW1fq" colab_type="code" outputId="d7f81b45-ccdf-4237-c266-cc137ce06a1a" colab={"base_uri": "https://localhost:8080/", "height": 1000}
from collections import Counter

def transform_face_image(img):
    return tf.image.rgb_to_grayscale(tf.image.resize(img, [48, 48])).numpy()

def genarete_emotions_counts(ds, num_steps):
    emotions = {}
    for i, (img_batch, filenames_batch) in enumerate(ds):
        if i == num_steps:
            break
        print(f"Step {i + 1}/{num_steps}")
        images = tf.image.convert_image_dtype(image_batch, tf.uint8).numpy()
        face_images = [
            [
                image[location[0]:location[2], location[3]:location[1]]
                for location in face_recognition.face_locations(image, model="cnn")
            ]
            for image in images
        ]
        for filename, face_images_per_img in zip(filenames_batch.numpy(), face_images):
            key = filename.decode().split("/")[-1][:-4]
            if face_images_per_img:
                models_input = np.array(list(map(transform_face_image, face_images_per_img)))
                predicted_emotions = list(map(lambda x: emotions_map_inv[x], np.argmax(face_recognition_model.predict(models_input), axis=1)))
                emotions[key] = Counter(predicted_emotions)
            else:
                emotions[key] = Counter()
    return emotions
emotions_dict = genarete_emotions_counts(ds, int(image_count/BATCH_SIZE) + 1)

# %% id="v5gpaB9VW1fu" colab_type="code" outputId="bc0f144b-90f7-4f53-a997-eae2c465d612" colab={"base_uri": "https://localhost:8080/", "height": 34}
len(emotions_dict), emotions_dict[list(emotions_dict.keys())[0]], list(emotions_dict.keys())[0]

# %% id="WSM0gdJqb-OR" colab_type="code" outputId="bf6da7fb-663e-4236-f8f1-a0a937fbcce9" colab={"base_uri": "https://localhost:8080/", "height": 102}
list(emotions_dict.keys())[:4], emotions_dict['https:i.ytimg.comviK0d_lOqB6tshqdefault']


# %% [markdown] id="7B7RL5feW1fx" colab_type="text"
# #### Constructing columns

# %% id="GZ-t87dnW1fy" colab_type="code" outputId="a988801c-6199-4362-e9bb-2648797b2d4c" colab={"base_uri": "https://localhost:8080/", "height": 1000}
def gen_column_value(thumbnail_url, emotion):
    filename = url2filename(thumbnail_url).replace("default.jpg", "hqdefault.jpg")[:-4]
    if filename in emotions_dict:
        counter = emotions_dict[filename]
        if emotion in counter:
            return counter[emotion]
    return 0

for emotion in emotions_map:
    emotion_count = df["thumbnail_link"].apply(lambda x: gen_column_value(x, emotion))
    if emotion_count.max() > 0:
        df[f"{emotion.lower()}_count"] = emotion_count
        sns.distplot(df[f"{emotion.lower()}_count"]).set_title(emotion)
        plt.show()


# %% [markdown] id="xx3_MDu_W1f1" colab_type="text"
# Conlusions
# > Only some of emotions are present at the images

# %% id="YNfGiO63yPut" colab_type="code" outputId="6f77a61b-7299-47de-e001-56f958f44a18" colab={"base_uri": "https://localhost:8080/", "height": 136}
df.columns

# %% id="NGcMSg5Z26wH" colab_type="code" colab={}
output_columns = ['thumbnail_link', 'thumbnail_ocr_embed', 'thumbnail_ocr_length',
       'angry_count', 'surprise_count', 'fear_count', 'happy_count']

# %% id="jM1_EyFeW1f1" colab_type="code" outputId="d2520033-a5d9-4714-afd2-1e3367dd00b9" colab={"base_uri": "https://localhost:8080/", "height": 297}
df[output_columns].describe()
# #### Emotions visualization


# %% id="mJMHIVQ9W1f5" colab_type="code" outputId="95d3ef0f-dc23-426b-de57-e0851ada03f4" colab={"base_uri": "https://localhost:8080/", "height": 704}
image_batch, filenames = next(iter(ds))
show_batch(image_batch.numpy(), [list(emotions_dict[str(filename).split("/")[-1][:-5]].keys()) for filename in filenames[:25].numpy()])

# %% id="rGmMty001hwR" colab_type="code" colab={}
df[output_columns].to_csv(os.path.join(path, "image_attributes_not_trending_nawrba.csv"), index=False)

# %% id="7Dn_OeVlcuc_" colab_type="code" colab={}
