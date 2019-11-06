import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from PIL import ImageFilter
from ..config import config
from .draw_text import text2bitmap

def text2bitmap_blurred(text, kernel_size=10):
    im = text2bitmap(text)
    blur_filter = ImageFilter.GaussianBlur(kernel_size)
    fim = im.filter(blur_filter)
    return fim

def text2bitmap_interleaved(text, stride=2):
    pass

def text2bitmap_flipped(text):
    im = text2bitmap(text)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im