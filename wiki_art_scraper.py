import os
import cv2
import numpy as np
import http.client
import urllib.request
from bs4 import BeautifulSoup

# Online art gallery url
wiki_art_path = 'https://artuk.org/discover/artworks/view_as/grid/search/keyword:landscape/page/500'
# Directory to save image database
img_dir = 'landscapes_dataset'
urls = []

# Parse the html
html = urllib.request.urlopen(wiki_art_path)
soup = BeautifulSoup(html)
for url in str(soup.findAll('img')).split('src="')[1:-5]:
    urls.append(url.partition('"')[0])

# Change the resolutions of the images in all urls
for i, url in enumerate(urls):
    urls[i] = url.replace('w300', 'w1200h1200')

# Download images
i = 0
for url in urls:
    i += 1
    filename = os.path.join(img_dir, f'l_{i}.jpg')
    try:
        urllib.request.urlretrieve(url, filename=filename)
        print('Successfully downloaded image number ' + str(i))
    # Catch exception with corrupted urls
    except http.client.InvalidURL:
        print('Failed to download image from url: ' + url)
        i -= 1

# List containing all our training data filenames
fnames = [os.path.join(img_dir, fname)
          for fname in os.listdir(img_dir)]

# This is the look of a painting that was failed to load
# or was not uploaded to the site. Let's delete all such images
unloaded = cv2.imread('unloaded.jpg')

for fname in fnames:
    img = cv2.imread(fname)
    try:
        if unloaded.shape == img.shape:
            difference = cv2.subtract(unloaded, img)
            b, g, r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                os.remove(fname)
                print("Image at path: " + fname + " removed")
    except AttributeError:
        print("Non-image file at path: " + fname)
