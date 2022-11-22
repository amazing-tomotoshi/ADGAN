# ADGAN/で実行
import os


dir_path = "data/img" 

for current_dir, sub_dirs, files_list in os.walk(dir_path): 
  print(files_list)
  # 色んな色が混ざってるからfor文で分ける
  # その後3:1のペアを組み合わせで作る
  # 長さ取得
  len = len(files_list)
  # 画像が3枚以下のものはスキップ
  if len < 4:
    continue

  num = ''
  files_list_color = []
  for i, file_name in enumerate(files_list):
    tmp_num = file_name[:2]
    if 1 == 0:
      num = tmp_num
      files_list_color.append(file_name)
      continue
    if tmp_num != num or i == len-1:
      # ここでfiles_list_colorの中でペア作って、csvに書き込む
      # files_list_colorは初期化
    if tmp_num == num:
      files_list_color.append(file_name)
      continue

    continue


  for file_name in files_list: 
    file_path = os.path.join(current_dir,file_name)
    print(file_path)
    cp_file_path = file_path[9:] #data/img/を削除
    cp_file_path = cp_file_path[::-1].replace("_", "", 1)[::-1] #最後の_を削除
    cp_file_path = cp_file_path.replace("id_", "id", 1)
    cp_file_path = cp_file_path.replace("/", "")
    cp_file_path = "data/deepfashion/img/fashion" + cp_file_path
    print(cp_file_path)
    # os.system('cp %s %s' %(file_path, cp_file_path))

