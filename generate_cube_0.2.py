import streamlit as st
import streamlit.components.v1 as components
import numpy as np
# import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import shutil
from glob import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import math
import matplotlib.pyplot as plt

from utils import CubeSide1, CubeSide2, CubeSide3, CubeSide4, CubeSide5, CubeSide6, \
                  display_3d_cube

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

st.set_page_config(layout="wide")

st.markdown("""
<style>
body {
    direction: RTL;
    unicode-bidi: bidi-override;
    # text-align: right;
}
</style>
""", unsafe_allow_html=True)

col_right, col_left = st.columns([0.7, 0.3])

graphics_dict = dict(
  bare_feet = Image.open(os.path.join(__location__, 'graphics/bare_feet_w200_h180.png'), 'r'),
  hand_by_hand = Image.open(os.path.join(__location__, 'graphics/hand_by_hand_w277_h288.png'), 'r'),
  ciconia_baby = Image.open(os.path.join(__location__, 'graphics/ciconia_baby_w180_h130.png'), 'r'),
  ruler = Image.open(os.path.join(__location__, 'graphics/ruler_w392_h40.png'), 'r'))

elements = {'first_name': 'בייבי',
            'last_name': 'בר דוד',
            'img_to_add2': graphics_dict.get('bare_feet'),
            'first_parent': 'מעוז',
            'second_parent': 'עפרי',
            'img_to_add3': graphics_dict.get('hand_by_hand'),
            'date_str': '2023-06-29',
            'time_str': '15:50',
            'img_to_add4': graphics_dict.get('ciconia_baby'),
            'num_kg': '3',
            'num_gram': '855',
            'num_cm': '49',
            'img_to_add5': graphics_dict.get('ruler'),
            'personal_greetings': 'באהבה דוד יאיר',
            }

with col_right:
  inputs_cols_first_row = st.columns(4)
  # st.markdown("<style>input, label {unicode-bidi:bidi-override; direction: RTL;}</style>", unsafe_allow_html=True)
  with inputs_cols_first_row[0]:
    elements['first_name'] = st.text_input('שם פרטי', 'אֵיתָן')
    elements['first_letter'] = elements['first_name'][0] if len(elements['first_name']) > 0 else 'א'
  with inputs_cols_first_row[1]:
    elements['last_name'] = st.text_input('שם משפחה', 'ישראלי')
  with inputs_cols_first_row[2]:
    elements['first_parent'] = st.text_input('שם האב', 'אבא')
  with inputs_cols_first_row[3]:
    elements['second_parent'] = st.text_input('שם האם', 'אמא')
  inputs_cols_second_row = st.columns(4)
  with inputs_cols_second_row[0]:
    elements['date_str'] = st.date_input("תאריך הלידה").strftime('%Y-%m-%d')
  with inputs_cols_second_row[1]:
    elements['time_str'] = st.time_input("שעת הלידה", step=60).strftime('%H:%M')
  with inputs_cols_second_row[2]:
    weight = float(st.number_input("משקל", step=0.001, format='%3f', min_value=0.0, max_value=9.999))
    elements['num_kg'] = str(math.floor(weight))
    elements['num_gram'] = str(int(round((weight - math.floor(weight)),3)*1000))
  with inputs_cols_second_row[3]:
    elements['num_cm'] = str(math.floor(float(st.number_input("גובה", step=1.0, format='%0f', min_value=0.0, max_value=99.9))))

  htmlstr = """<script>var elements = window.parent.document.querySelectorAll('input'), i ;
                       elements[4].style.direction = 'LTR'; 
                       elements[5].style.direction = 'LTR'; 
                       elements[6].style.direction = 'LTR';
                       elements[7].style.direction = 'LTR';
              </script>  """

  components.html(f"{htmlstr}", height=0, width=0)


w = 803
h = 803
w = 320 # 240
h = w
cube_sides_dict = {1:CubeSide1(elements=elements, w_dis=w, h_dis=h),
                    2:CubeSide2(elements=elements, w_dis=w, h_dis=h),
                    3:CubeSide3(elements=elements, w_dis=w, h_dis=h),
                    4:CubeSide4(elements=elements, w_dis=w, h_dis=h),
                    5:CubeSide5(elements=elements, w_dis=w, h_dis=h),
                    6:CubeSide6(elements=elements, w_dis=w, h_dis=h),
}




with col_right:
  # st.markdown("<style>img {text-align: center;}</style>", unsafe_allow_html=True)
  images_cols = st.columns(4)
  col_i = 0
  for img_i in range(2,6):
    with images_cols[col_i]:
      st.image(cube_sides_dict[img_i].img_dis.display_img(border_dict={'border_color': (200,200,200,255)})) # caption=2
      hide_img_fs = '''
      <style>
      button[title="View fullscreen"]{
          visibility: hidden;}
      </style>
      '''
      st.markdown(hide_img_fs, unsafe_allow_html=True)
      col_i += 1
      if col_i >= len(images_cols):
        col_i = 0
  cube6_cols = st.columns(2)
  with cube6_cols[0]:
    elements['personal_greetings'] = st.text_area('ברכה אישית', 'באהבה גדולה', help='לא יותר משלוש שורות, אפשר להוסיף רווחים בשביל יישור שונה')
    cube_sides_dict[6] = CubeSide6(elements=elements, w_dis=w, h_dis=h)
  with cube6_cols[1]:
    st.image(cube_sides_dict[6].img_dis.display_img(border_dict={'border_color': (200,200,200,255)})) # caption=2
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)


img_siz_for_fig = 240
cube_sides_dict_for_fig = {1:CubeSide1(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
                    2:CubeSide2(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
                    3:CubeSide3(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
                    4:CubeSide4(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
                    5:CubeSide5(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
                    6:CubeSide6(elements=elements, w_dis=img_siz_for_fig, h_dis=img_siz_for_fig),
}


imgs_dict = {k: v.img_dis.img for k, v in cube_sides_dict_for_fig.items()}

fig = display_3d_cube(imgs_dict, img_siz_for_fig, img_siz_for_fig)


with col_left:
  st.button(label="תצוגת תלת מימד", key="display_3d_cube")
  if st.session_state.get("display_3d_cube"):
    st.plotly_chart(fig, use_container_width=True)