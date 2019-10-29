import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from ..config import config



def step_char_wise(im, **kwargs):    
    nchar = kwargs.get("nchar", 1)
    char_width = config.FONT_SIZE
    xoffset = round(kwargs.get("xoffset", 0) * char_width)
    step_size = char_width * nchar
    step_overlap = kwargs.get("step_overlap", False)
    if step_overlap:
        steps = list(range(xoffset, im.size[0], char_width))
    else:
        steps = list(range(xoffset, im.size[0], step_size))    
    return steps, step_size
    
def measure_text(text):    
    font_size = config.FONT_SIZE
    font = ImageFont.truetype(config.DEFAULT_FONT_PATH, font_size)    
    # measure text    
    txtw, txth = font.getsize(text)
    return txtw, txth

def text2bitmap(text):    
    font_size = config.FONT_SIZE
    font = ImageFont.truetype(config.DEFAULT_FONT_PATH, font_size)    

    im_dim = measure_text(text)
    # "L" for a 8bit bitmap    
    im = Image.new("L", im_dim)
    draw = ImageDraw.Draw(im)    
    draw.text((0,0), text, (255,), font=font)    
    return im

def get_filtered_shape(img, coord):
    w, h = img.shape
    x, y, sw, sh = coord 
    
    if x > w or y > h:
        raise ValueError(f"{(x, y)} not in img dim {(w, h)}")
    if x + sw > w:
        sw = w - x
        
    if y + sh > h:
        sh = h - y
    
    return x, y, sw, sh

def spotlight_rectangle(img, coord, **kwarg):
    x, y, sw, sh = get_filtered_shape(img, coord)
    padw = 0; padh = 0
    if sw < coord[2]:
        padw = coord[2] - sw
    if sh < coord[3]:
        padh = coord[3] - sh
    
    simg = img[x:(x+sw), y:(y+sh)]
    fimg = np.pad(simg, ((0, padw), (0, padh)), "constant")
    
    return fimg