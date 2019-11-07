import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from ..config import config
from .draw_text import text2bitmap

def text2bitmap_blurred(text, kernel_size=2, im_dim=None):
    im = text2bitmap(text, im_dim)
    blur_filter = ImageFilter.GaussianBlur(kernel_size)
    fim = im.filter(blur_filter)
    return fim

def text2bitmap_interleaved(text, stride=2, im_dim=None):
    im = text2bitmap(text, im_dim)
    w, h = im.size
    for w_i in range(0, w, stride):
        for h_i in range(0, h, stride):
            im.putpixel((w_i, h_i), 0)
    return im

def text2bitmap_random(text, stride=2, im_dim=None, seed=None):
    rnd = np.random.RandomState(seed)    
    im = text2bitmap(text, im_dim)
    w, h = im.size
    for w_i in range(0, w, stride):
        for h_i in range(0, h, stride):
            if rnd.uniform() > 0.1: continue
            im.putpixel((w_i, h_i), 0)
    return im

def text2bitmap_flipped(text, im_dim=None):
    im = text2bitmap(text, im_dim)
    im = im.rotate(0, translate=(0, -12))
    im = im.transpose(Image.FLIP_TOP_BOTTOM)    
    return im