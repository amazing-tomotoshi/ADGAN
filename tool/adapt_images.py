# ADGAN/で実行
import os


dir_path = "data/img" 

for current_dir, sub_dirs, files_list in os.walk(dir_path): 
  for file_name in files_list: 
    file_path = os.path.join(current_dir,file_name)
    print(file_path)
    cp_file_path = file_path[9:] #data/img/を削除
    cp_file_path = cp_file_path[::-1].replace("_", "", 1)[::-1] #最後の_を削除
    cp_file_path = cp_file_path.replace("id_", "id", 1)
    cp_file_path = cp_file_path.replace("/", "")
    cp_file_path = "data/deepfashion/img/fashion" + cp_file_path
    print(cp_file_path)
    os.system('cp %s %s' %(file_path, cp_file_path))

