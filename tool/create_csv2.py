# ADGAN/で実行
import os
import csv
import itertools
import random
import re

dir_path = "data/deepfashion/test" 

with open('./data/deepfashion/fashion-resize-pairs-test2.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(['from1', 'from2', 'from3', 'to'])

  files = sorted(os.listdir(dir_path))

  num = ''
  files_list_color = []
  for i, file_name in enumerate(files):
    tmp_num = re.search(r'\d{10}', file_name).group()
    print(tmp_num)
    if num == '':
      num = tmp_num
      files_list_color.append(file_name)
      continue
    if tmp_num == num:
      files_list_color.append(file_name)
      if i != len(files)-1:
        continue
    if tmp_num != num or i == len(files)-1:
      # ここでfiles_list_colorの中でペア作って、csvに書き込む
      # files_list_colorは初期化
      if len(files_list_color) < 4:
        files_list_color = []
        files_list_color.append(file_name)
        num = tmp_num
        continue
      elif len(files_list_color) == 4:
        print('flag!!')
        writer.writerow([files_list_color[0], files_list_color[1], files_list_color[2], files_list_color[3]])
        writer.writerow([files_list_color[1], files_list_color[2], files_list_color[3], files_list_color[0]])
        writer.writerow([files_list_color[2], files_list_color[3], files_list_color[0], files_list_color[1]])
        writer.writerow([files_list_color[3], files_list_color[0], files_list_color[1], files_list_color[2]])
        files_list_color = []
        files_list_color.append(file_name)
        num = tmp_num
      else:
        for v in itertools.combinations(files_list_color, 4):
          v = list(v)
          random.shuffle(v)
          writer.writerow(v)
          files_list_color = []
          files_list_color.append(file_name)
          num = tmp_num


