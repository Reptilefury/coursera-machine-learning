import fastbook
import matplotlib
fastbook.setup_book()
import tkinter

from fastbook import *
from fastai.vision.all import *
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image




a = 1
b = a + 1
c = a + b + 1
d = a + b + c + 1

print(a,b,c,d)
plt.plot([a,b,c,d])

#plt.show()
Image.open(image_cat())





from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
path, get_image_files(path), valid_pct=0.2, seed=42,
label_func=is_cat, item_tfms=Resize(224))
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
#print(test)
#plt.savefig("graph.png")
