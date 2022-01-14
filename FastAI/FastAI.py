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

test = Image.open(image_cat())

print(test)
#plt.savefig("graph.png")
