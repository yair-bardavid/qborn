import streamlit as st
import numpy as np
# import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import shutil
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import matplotlib.pyplot as plt


FONTS_DICT = dict(aram='ARAM.TTF',
                  alef='Alef-Regular.ttf',
                  cafe='CAFERG_.TTF',
                  californian='CALIFB.TTF',
                  dana='DanaYadAlefAlefAlef-Normal.otf',
                  firamono='FiraMono-Regular.ttf',
                  freeserif='FreeSerif.ttf',
                  gill='GIL_____.TTF',
                  keren='Guttman_Keren-Normal.ttf',
                  tzvi='TZVIM.TTF',
                  )

def mm2pix(mm, PPI=300):
  return round(mm / ( 25.4 / PPI ))

def pix2mm(pix, PPI=300):
  return round(pix * ( 25.4 / PPI ))

def detect_is_text_in_heb(text):
  heb_abc = set('אבגדהוזחטיכלמנסעפצקרשתךםןףץ')
  if len(text) > 1:
    for char in text:
      if char in heb_abc:
        return True
  else:
    if text in heb_abc:
        return True

  return False

def is_char_a_nikud(char):
  # http://www.nashbell.com/technology/he-unicode.php
  nikud_unicode_range = set(range(1425, 1480)) - set([1470, 1472, 1475, 1478])
  if ord(char) in nikud_unicode_range:
    return True
  return False


class CusImg(object):
  def __init__(self, w, h):
    self.w = w
    self.h = h
    self.img = Image.new('RGBA', (self.w, self.h), (255, 255, 255, 0))

  def display_img(self, erode=None, dilate=None, border_dict=None):
    if erode is not None:
      self.erode(erode)
    if dilate is not None:
      self.dilate(dilate)

    if border_dict is None:
      img_to_show = self.img
    else:
      img_to_show = self.get_im_with_border(
          border_width=border_dict.get('border_width', 1),
          border_color=border_dict.get('border_color', (0,0,0,255))
        )
    # plt.imshow(img_to_show)
    # plt.axis('off')
    # plt.show()
    return img_to_show

  def erode(self, val):
    self.img = self.img.filter(ImageFilter.MinFilter(val))

  def dilate(self, val):
    self.img = self.img.filter(ImageFilter.MaxFilter(val))


  def add_text(self, xpos, ypos, message, font_name, font_size,
               fill=(0, 0, 0), kerning=0, reduce_size_factor=0.95,
               rotate=0):
    text_dir = 'rtl' if detect_is_text_in_heb(message) else 'ltr'
    len_message_ignore_nikud = sum([not is_char_a_nikud(ch) for ch in message])
    w, h = self.w, self.h
    if rotate % 180 == 90:
      w, h = h, w
    if rotate % 360 == 90:
      xpos, ypos = 1-ypos, xpos
    if rotate % 360 == 180:
      xpos, ypos = 1-xpos, 1-ypos
    if rotate % 360 == 270:
      xpos, ypos = ypos, 1-xpos
    img_for_draw = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    imgDraw = ImageDraw.Draw(img_for_draw)

    font = ImageFont.truetype(font_name, size=font_size)
    x1, y1, x2, y2 = imgDraw.textbbox((0,0), message, font=font)
    w_t = x2 - x1 + (len_message_ignore_nikud-1)*kerning

    while w_t >= w * reduce_size_factor:
      font_size *= reduce_size_factor
      font = ImageFont.truetype(font_name, size=int(font_size))
      x1, y1, x2, y2 = imgDraw.textbbox((0,0), message, font=font)
      w_t = x2 - x1 + (len_message_ignore_nikud-1)*kerning

    h_t = y2 - y1

    x_t = (w - w_t) * xpos - x1
    y_t = (h - h_t) * ypos - y1

    text_slices = []
    i = 0
    while i < len(message):
      j = 0
      if i+j+1 < len(message):
        while is_char_a_nikud(message[i+j+1]) or message[i+j+1].isdigit():
          j += 1
          if i+j+1 >= len(message):
            break
      text_slices.append(message[i:i+j+1])
      i += j + 1

    text_slices = text_slices[::-1] if text_dir == 'rtl' else text_slices
    for text_slice in text_slices:
      imgDraw.text((x_t, y_t), text_slice, font=font, fill=fill)
      xx1, _, xx2, _ = imgDraw.textbbox((0,0), text_slice, font=font)
      x_t += xx2 - xx1 + kerning

    img_for_draw = img_for_draw.rotate(rotate,  expand=1)
    self.img.paste(img_for_draw, (0,0), img_for_draw)

  def add_image(self, xpos, ypos, img_to_add, reduce_size_factor=None):
    if reduce_size_factor is None:
      reduce_size_factor = self.w / 803.0
    w_i_orig, h_i_orig = img_to_add.size
    img_to_add = img_to_add.resize((int(w_i_orig*reduce_size_factor),int(h_i_orig*reduce_size_factor)), Image.Resampling.LANCZOS)
    w_i, h_i = img_to_add.size
    x = int((self.w - w_i) * xpos)
    y = int((self.h - h_i) * ypos)
    self.img.paste(img_to_add, (x, y), img_to_add)


  def add_centered_circle(self, offset_factor=0.05, fill=(0, 0, 0)):
    imgDraw = ImageDraw.Draw(self.img)
    min_dim = min(self.w, self.h)
    dim_diff = max(self.w, self.h) - min_dim
    is_w_greater = self.w > self.h
    xy = (min_dim*offset_factor + (dim_diff/2)*is_w_greater,
          min_dim*offset_factor + (dim_diff/2)*(1-is_w_greater),
          min_dim - (min_dim*offset_factor) + (dim_diff/2)*is_w_greater,
          min_dim - (min_dim*offset_factor) + (dim_diff/2)*(1-is_w_greater))
    imgDraw.ellipse(xy, fill=fill)

  def get_im_with_border(self, border_width=1, border_color=(0, 0, 0, 255)):
    img_a = np.array(self.img)
    img_a[:border_width,:,:] = border_color
    img_a[:,:border_width,:] = border_color
    img_a[-border_width:,:,:] = border_color
    img_a[:,-border_width:,:] = border_color
    return Image.fromarray(img_a)


class CubeSide(object):
  def __init__(self, elements=None, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    self.w_dis = w_dis
    self.h_dis = h_dis
    self.w_exp = w_dis if w_exp is None else w_exp
    self.h_exp = h_dis if h_exp is None else h_exp
    self.img_dis = CusImg(self.w_dis, self.h_dis)
    self.img_exp = CusImg(self.w_exp, self.h_exp)
    if elements is None:
      elements = {}
    self.elements = elements
    self.default_heb_strings = dict(parents_title='ההורים המאושרים', bet='ב',
                                    kg='קילוגרם', gram='גרם', cm='ס"מ')
    self.eng_week_day2heb = dict(Sunday='יום ראשון', Monday='יום שני',
                                 Tuesday='יום שלישי', Wednesday='יום רביעי',
                                 Thursday='יום חמישי', Friday='יום שישי',
                                 Saturday='יום שבת',)
    self.month2heb = {1: 'ינואר', 2: 'פברואר', 3: 'מרץ', 4: 'אפריל',
                      5: 'מאי', 6: 'יוני', 7: 'יולי', 8: 'אוגוסט',
                      9: 'ספטמבר', 10: 'אוקטובר', 11: 'נובמבר', 12: 'דצמבר'}

  def add_text(self, xpos, ypos, message, font_name, font_size,
               fill=(0, 0, 0), kerning=0, rotate=0):
    self.img_dis.add_text(xpos, ypos, message, font_name, font_size, fill=fill, kerning=kerning, rotate=rotate)
    self.img_exp.add_text(xpos, ypos, message, font_name, font_size, fill=fill, kerning=kerning, rotate=rotate)

  def add_image(self, xpos, ypos, img_to_add):
    self.img_dis.add_image(xpos, ypos, img_to_add)
    self.img_exp.add_image(xpos, ypos, img_to_add)


class CubeSide1(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    self.img_dis.add_centered_circle(offset_factor=0.05, fill='black')
    self.img_exp.add_centered_circle(offset_factor=0.05, fill='black')

    txt_iden = 'first_letter'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('keren') if detect_is_text_in_heb(self.elements.get(txt_iden)) else FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.7 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.5, 0.5, self.elements[txt_iden], font_name, font_size, fill='white')


class CubeSide2(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    txt_iden = 'first_name'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('aram'))
    font_size = int(self.h_dis*0.2 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.5, 0.2, self.elements[txt_iden], font_name, font_size, fill='black', kerning=int(font_size*0.15))

    txt_iden = 'last_name'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('aram'))
    font_size = int(self.h_dis*0.14 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.5, 0.55, self.elements[txt_iden], font_name, font_size, fill='black', kerning=int(font_size*0.15))

    self.add_image(0.85, 0.9, self.elements['img_to_add2'])



class CubeSide3(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    txt_iden = 'parents_title'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('cafe'))
    font_size = int(self.h_dis*0.08 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.5, 0.1, self.elements.get(txt_iden, self.default_heb_strings[txt_iden]),
                  font_name, font_size, fill='black', kerning=int(font_size*0.15))

    max_len_parent = max(sum([not is_char_a_nikud(ch) for ch in self.elements.get('first_parent')]),
                         sum([not is_char_a_nikud(ch) for ch in self.elements.get('second_parent')]))
    parent_size_resize = 5.0/max(5, max_len_parent)

    txt_iden = 'first_parent'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('tzvi') if detect_is_text_in_heb(self.elements.get(txt_iden)) else FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.19 * parent_size_resize * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.7, 0.32, self.elements.get(txt_iden),
                  font_name, font_size, fill='black', kerning=int(font_size*0.05))

    txt_iden = 'connector'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('gill'))
    font_size = int(self.h_dis*0.085 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.5, 0.5, self.elements.get(txt_iden, '&'),
                  font_name, font_size, fill='black', kerning=int(font_size*0.05))

    txt_iden = 'second_parent'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('tzvi') if detect_is_text_in_heb(self.elements.get(txt_iden)) else FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.19 * parent_size_resize * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.3, 0.68, self.elements.get(txt_iden),
                  font_name, font_size, fill='black', kerning=int(font_size*0.05))

    self.add_image(0.92, 0.92, self.elements['img_to_add3'])


class CubeSide4(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    date_str = self.elements.get('date_str')
    time_str = self.elements.get('time_str')
    date_datetime = datetime.strptime(date_str, '%Y-%m-%d')
    time_datetime = datetime.strptime(time_str, '%H:%M')
    year = date_datetime.year
    month = date_datetime.month
    day = date_datetime.day
    # hour = time_datetime.hour
    # minute = time_datetime.minute
    hour_minute = time_datetime.strftime("%H:%M %p")

    heb_week_day = self.eng_week_day2heb.get(date_datetime.strftime('%A'))
    heb_month = self.month2heb.get(month)

    txt_iden = 'heb_week_day'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('tzvi'))
    font_size = int(self.h_dis*0.12 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.17, 0.5, heb_week_day, font_name, font_size,
                  fill='black', kerning=int(font_size*0.05), rotate=90)

    txt_iden = 'day_month'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.1 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.68, 0.6, f'{day} {self.default_heb_strings.get("bet")}{heb_month}', font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    txt_iden = 'year'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('freeserif'))
    font_size = int(self.h_dis*0.2 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.68, 0.8, str(year), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    txt_iden = 'hour_minute'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('freeserif'))
    font_size = int(self.h_dis*0.12 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.68, 0.2, str(hour_minute), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    self.add_image(0.68, 0.4, self.elements['img_to_add4'])


class CubeSide5(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    txt_iden = 'num_kg'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('freeserif'))
    font_size = int(self.h_dis*0.4 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.3, 0.25, self.elements.get(txt_iden), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    txt_iden = 'kg'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.08 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.16, 0.25, self.elements.get(txt_iden, self.default_heb_strings[txt_iden]), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05), rotate=90)

    txt_iden = 'num_gram'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('freeserif'))
    font_size = int(self.h_dis*0.15 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.75, 0.22, self.elements.get(txt_iden), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    txt_iden = 'gram'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.08 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.72, 0.38, self.elements.get(txt_iden, self.default_heb_strings[txt_iden]), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05), rotate=0)

    txt_iden = 'num_cm'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('freeserif'))
    font_size = int(self.h_dis*0.18 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.63, 0.7, self.elements.get(txt_iden), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05))

    txt_iden = 'cm'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.08 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    self.add_text(0.38, 0.72, self.elements.get(txt_iden, self.default_heb_strings[txt_iden]), font_name, font_size,
                  fill='black', kerning=int(font_size*0.05), rotate=0)

    self.add_image(0.5, 0.85, self.elements['img_to_add5'])

class CubeSide6(CubeSide):
  def __init__(self, elements, w_dis=803, h_dis=803, w_exp=None, h_exp=None):
    super().__init__(elements, w_dis, h_dis, w_exp, h_exp)

    txt_iden = 'personal_greetings'
    font_name = self.elements.get(f'{txt_iden}_font_name', FONTS_DICT.get('cafe') if detect_is_text_in_heb(self.elements.get(txt_iden)) else FONTS_DICT.get('alef'))
    font_size = int(self.h_dis*0.13 * self.elements.get(f'{txt_iden}_font_resize', 1.0))
    greeting_text = self.elements.get(txt_iden, '')
    y_spacer = 0.27
    lines = greeting_text.split('\n')
    for i, line in enumerate(lines):
      y_spacer_factor = 1.0
      if i > 0:
        y_spacer_factor = min(y_spacer_factor, (13.0/len(lines[i-1])))
      self.add_text(0.5, 0.15 + i*y_spacer*y_spacer_factor, line,
                    font_name, font_size, fill='black', kerning=int(font_size*0.05))

    # self.add_image(0.5, 0.15, self.elements['img_to_add6'])

def display_3d_cube(imgs_dict, w, h):
  pl_grey =[[0.0, 'rgb(0, 0, 0)'], [1.0, 'rgb(255, 255, 255)']]
  x = np.linspace(0,w, w)
  y = np.linspace(0, h, h)
  X, Y = np.meshgrid(x,y)

  surfs = []

  gs_im = np.array(imgs_dict[1].convert('L'))
  surf = go.Surface(x=X, y=Y, z=X-X+h,
                    surfacecolor=np.fliplr(gs_im),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)

  gs_im = np.array(imgs_dict[2].convert('L'))
  surf = go.Surface(x=X, y=X-X+h, z=Y,
                    surfacecolor=np.rot90(gs_im, k=2),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)

  gs_im = np.array(imgs_dict[3].convert('L'))
  surf = go.Surface(x=Y-Y+w, y=Y, z=X,
                    surfacecolor=np.rot90(gs_im, k=-1),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)

  gs_im = np.array(imgs_dict[4].convert('L'))
  surf = go.Surface(x=X, y=X-X, z=Y,
                    surfacecolor=np.rot90(np.fliplr(gs_im), k=2),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)

  gs_im = np.array(imgs_dict[5].convert('L'))
  surf = go.Surface(x=Y-Y, y=Y, z=X,
                    surfacecolor=np.rot90(np.fliplr(gs_im), k=-1),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)

  gs_im = np.array(imgs_dict[6].convert('L'))
  surf = go.Surface(x=X, y=Y, z=X-X,
                    surfacecolor=np.rot90(gs_im, k=2),
                    colorscale=pl_grey,
                    showscale=False)
  surfs.append(surf)


  layout = go.Layout(
          title={'text' : "תסובבו אותי",'x':0.5,'xanchor': 'center'},
          font_family='Balto',
          width=500,
          height=500,
          scene=dict(
              xaxis_visible=False,
              yaxis_visible=False,
              zaxis_visible=False,
              aspectratio=dict(x=1,
                              y=1,
                              z=1
                              )
            ),
          hovermode=False
          )

  fig = go.Figure(data=surfs, layout=layout)
  fig_show_config = {'displayModeBar': True,
                    'displaylogo': False}
  # fig.show(config=fig_show_config)
  return fig
