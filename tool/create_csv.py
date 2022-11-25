# ADGAN/で実行
import os
import csv
import itertools
import random

dir_path = "data/img" 

def changeName(current_dir, file_name):
  file_path = os.path.join(current_dir,file_name)
  file_path = file_path[9:] #data/img/を削除
  file_path = file_path[::-1].replace("_", "", 1)[::-1] #最後の_を削除
  file_path = file_path.replace("id_", "id", 1)
  file_path = file_path.replace("/", "")
  file_path = "fashion" + file_path
  return file_path

with open('./data/deepfashion/fashion-resize-pairs-test2.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(['from1', 'from2', 'from3', 'to'])

  for current_dir, sub_dirs, files_list in os.walk(dir_path): 
    print(files_list)
    # 色んな色が混ざってるからfor文で分ける
    # その後3:1のペアを組み合わせで作る
    # 長さ取得
    length = len(files_list)
    # 画像が3枚以下のものはスキップ
    if length < 4:
      continue

    num = ''
    files_list_color = []
    for i, file_name in enumerate(files_list):
      tmp_num = file_name[:2]
      if i == 0:
        num = tmp_num
        files_list_color.append(changeName(current_dir, file_name))
        continue
      if tmp_num != num or i == length-1:
        # ここでfiles_list_colorの中でペア作って、csvに書き込む
        # files_list_colorは初期化
        if len(files_list_color) < 4:
          continue
        elif len(files_list_color) == 4:
          writer.writerow([files_list_color[0], files_list_color[1], files_list_color[2], files_list_color[3]])
          writer.writerow([files_list_color[1], files_list_color[2], files_list_color[3], files_list_color[0]])
          writer.writerow([files_list_color[2], files_list_color[3], files_list_color[0], files_list_color[1]])
          writer.writerow([files_list_color[3], files_list_color[0], files_list_color[1], files_list_color[2]])
          files_list_color = []
        else:
          for v in itertools.combinations(files_list_color, 4):
            v = list(v)
            random.shuffle(v)
            writer.writerow(v)
            files_list_color = []
      if tmp_num == num:
        files_list_color.append(changeName(current_dir, file_name))
        continue


