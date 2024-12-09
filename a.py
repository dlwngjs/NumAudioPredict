import os
import shutil
import re

# 상위 폴더 경로 설정
parent_folder = './train_path'

# 상위 폴더 경로가 존재하는지 확인
if not os.path.exists(parent_folder):
    print(f"Error: The folder {parent_folder} does not exist!")
else:
    print(f"Folder {parent_folder} exists!")

# 0~9까지의 하위 폴더를 생성
for i in range(10):
    subfolder_path = os.path.join(parent_folder, str(i))
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        print(f"Created folder: {subfolder_path}")
    else:
        print(f"Folder {subfolder_path} already exists!")

# 상위 폴더 내의 파일들을 확인하여, .wav 파일을 해당 폴더로 이동
files = os.listdir(parent_folder)
print(f"Files in {parent_folder}: {files}")  # 상위 폴더 내 파일 목록 출력

for file_name in files:
    file_path = os.path.join(parent_folder, file_name)
    
    # 파일인지 확인하고, .wav로 끝나는 파일만 처리 (확장자 체크를 더 정확하게)
    if os.path.isfile(file_path):
        # 파일 이름을 소문자로 변환하고 확장자 확인
        file_name_lower = file_name.lower()
        
        if file_name_lower.endswith('.wav'):  # 대소문자 구분없이 .wav 파일만 처리
            print(f"Processing file: {file_name}")  # 처리 중인 파일 출력

            # 파일 이름에서 마지막 숫자 추출 (예: 'f10.wav' -> 0, 'm42.wav' -> 2)
            match = re.search(r'(\d)\.wav$', file_name)
            
            if match:
                label = match.group(1)  # 정규 표현식에서 추출한 마지막 숫자
                print(f"Extracted label: {label}")  # 라벨 추출 확인
                
                if label.isdigit() and int(label) in range(10):  # 0~9만 처리
                    target_folder = os.path.join(parent_folder, label)
                    target_path = os.path.join(target_folder, file_name)

                    # 이동할 경로 출력 (디버깅용)
                    print(f"Moving file: {file_name} from {file_path} to {target_path}")
                    
                    try:
                        # 파일 이동
                        shutil.move(file_path, target_path)
                        print(f"Successfully moved {file_name} to {target_folder}")
                    except Exception as e:
                        print(f"Error moving {file_name}: {e}")
            else:
                print(f"No valid label found for {file_name}")
        else:
            print(f"Skipping non-wav file: {file_name}")  # .wav가 아닌 파일 처리
    else:
        print(f"Skipping non-file: {file_name}")  # 디렉토리나 기타 파일 처리


file_dict = {}

for i in range(10):
    subfolder_path = os.path.join(parent_folder, str(i))
    
    # 해당 하위 폴더가 존재하는 경우
    if os.path.isdir(subfolder_path):
        file_list = []
        
        # 하위 폴더 내의 파일들을 확인
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            
            # 파일인 경우에만 리스트에 추가
            if os.path.isfile(file_path):
                # 부모 폴더를 포함한 파일 경로 저장
                full_file_path = os.path.join(parent_folder, str(i), file_name)
                file_list.append(full_file_path)
        
        # 하위 폴더에 파일이 있으면 딕셔너리에 저장
        if file_list:
            file_dict[str(i)] = file_list

# 결과 출력 (예: 라벨과 그에 해당하는 파일 목록)
print(file_dict)