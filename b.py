import os
import re

def label_files_by_last_number(folder_path):
    file_dict = {}

    # 폴더 내의 모든 파일을 읽어옴
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 파일인 경우에만 처리
        if os.path.isfile(file_path):
            # 파일명에서 마지막 숫자를 추출하는 정규 표현식
            match = re.search(r'(\d+)(?=\.\w+$)', file_name)
            
            if match:
                last_number = int(match.group(1)) % 10  # 마지막 숫자를 기준으로 0~9로 라벨링
                if last_number not in file_dict:
                    file_dict[last_number] = []
                file_dict[last_number].append(file_path)  # 파일 경로를 저장
    
    return file_dict

# 사용 예시
folder_path = './path/test'  # 파일들이 있는 폴더 경로를 설정하세요.
file_dict = label_files_by_last_number(folder_path)

# 결과 출력
print(file_dict)
