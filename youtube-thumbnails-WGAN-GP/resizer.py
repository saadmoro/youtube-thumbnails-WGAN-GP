from PIL import Image
import os.path

path_in = 'thumbs/'
path_out = 'resized/'
size = (100, 100)

dirs = os.listdir(path_in)
for item in dirs:
    if os.path.isfile(path_in + item):
        im = Image.open(path_in + item)
        f, e = os.path.split(path_in + item)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(f'{path_out}/{e}', 'JPEG', quality=90)
