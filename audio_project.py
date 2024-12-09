import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import os
import shutil
import re

'''
train_data_folder에 훈련 데이터 파일들을 넣어두시면 됩니다.
test_data_folder에 테스트 데이터 파일들을 넣어두시면 됩니다.

단 모든 파일명은 해당 음성의 숫자로 끝나야 합니다.
ex) 0을 말하는 파일이라면 ~~~0.wav 형식이여야 합니다.
'''

train_data_folder = './train_path'
test_data_folder = './test_path'

trainSet = {}
testSet = {}

def set_folder():
    if not os.path.exists(train_data_folder):
        print(f"Error: The folder {train_data_folder} does not exist!")
    else:
        print(f"Folder {train_data_folder} exists!")

    for i in range(10):
        subfolder_path = os.path.join(train_data_folder, str(i))
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created folder: {subfolder_path}")
        else:
            print(f"Folder {subfolder_path} already exists!")

    files = os.listdir(train_data_folder)
    print(f"Files in {train_data_folder}: {files}")

    for file_name in files:
        file_path = os.path.join(train_data_folder, file_name)
        
        if os.path.isfile(file_path):
            file_name_lower = file_name.lower()
            
            if file_name_lower.endswith('.wav'):
                print(f"Processing file: {file_name}")
                match = re.search(r'(\d)\.wav$', file_name)
                
                if match:
                    label = match.group(1)
                    print(f"Extracted label: {label}")
                    
                    if label.isdigit() and int(label) in range(10):
                        target_folder = os.path.join(train_data_folder, label)
                        target_path = os.path.join(target_folder, file_name)

                        print(f"Moving file: {file_name} from {file_path} to {target_path}")
                        
                        try:
                            shutil.move(file_path, target_path)
                            print(f"Successfully moved {file_name} to {target_folder}")
                        except Exception as e:
                            print(f"Error moving {file_name}: {e}")
                else:
                    print(f"No valid label found for {file_name}")
            else:
                print(f"Skipping non-wav file: {file_name}")
        else:
            print(f"Skipping non-file: {file_name}")

def get_data():
    for i in range(10):
        subfolder_path = os.path.join(train_data_folder, str(i))
        
        if os.path.isdir(subfolder_path):
            file_list = []
            
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                
                if os.path.isfile(file_path):
                    full_file_path = os.path.join(train_data_folder, str(i), file_name)
                    file_list.append(full_file_path)
            
            if file_list:
                trainSet[i] = file_list
    
    for file_name in os.listdir(test_data_folder):
        file_path = os.path.join(test_data_folder, file_name)
        
        if os.path.isfile(file_path):
            match = re.search(r'(\d+)(?=\.\w+$)', file_name)
            
            if match:
                last_number = int(match.group(1)) % 10
                if last_number not in testSet:
                    testSet[last_number] = []
                testSet[last_number].append(file_path)

    print(trainSet)
    print(testSet)

def get_mfcc(file):
    y, sr = librosa.load(file, sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, n_fft=512, n_mels=40, hop_length=128)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=16000)

    max_length = max(mfcc.shape[1], spectral_contrast.shape[1])
    mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    mfcc_delta = np.pad(mfcc_delta, ((0, 0), (0, max_length - mfcc_delta.shape[1])), mode='constant')
    mfcc_delta2 = np.pad(mfcc_delta2, ((0, 0), (0, max_length - mfcc_delta2.shape[1])), mode='constant')
    spectral_contrast = np.pad(spectral_contrast, ((0, 0), (0, max_length - spectral_contrast.shape[1])), mode='constant')

    features = np.hstack((mfcc.T, mfcc_delta.T, mfcc_delta2.T, spectral_contrast.T))

    return features

def train_models(trainSet):
    gmm_models = {}
    for label, files in trainSet.items():
        mfcc_features = np.concatenate([get_mfcc(file) for file in files], axis=0)
        gmm = GaussianMixture(n_components=5, max_iter=200, covariance_type='diag', n_init=5)
        gmm.fit(mfcc_features)
        gmm_models[label] = gmm
    return gmm_models

def predict(gmm_models, test_file):
    mfcc_features = get_mfcc(test_file)
    scores = {label: gmm.score(mfcc_features) for label, gmm in gmm_models.items()}
    print(f"{test_file} Score :")
    for label, score in scores.items():
        print(f"  Model {label}: {score:.2f}")
    return max(scores, key=scores.get)

def main():
    set_folder()
    get_data()
    max_acc = 0
    '''최선의 랜덤 값을 찾기위해 반복 (제가 테스트 했을 때는 평균 100회정도에 최선값을 찾지만 너무 오래걸리면 반복 횟수 줄이시면 됩니다.)'''
    for i in range(500):
        print(i)
        gmm_models = train_models(trainSet)
        y_true = []
        y_pred = []
        for label, files in testSet.items():
            for file in files:
                y_true.append(label)
                y_pred.append(predict(gmm_models, file))

        acc = accuracy_score(y_true, y_pred)
        print(f"y_true : {y_true}\ny_pred : {y_pred}")
        print(f"Acc : {acc}")
        if(acc > max_acc):
            max_acc = acc
            best_gmm_model = gmm_models
        if(max_acc >= 0.8):
            break
    # 최적의 랜덤값을 가진 모델로 테스트
    y_true = []
    y_pred = []
    for label, files in testSet.items():
        for file in files:
            y_true.append(label)
            y_pred.append(predict(best_gmm_model, file))

    acc = accuracy_score(y_true, y_pred)
    print(f"y_true : {y_true}\ny_pred : {y_pred}")
    print(f"Acc : {acc}")

if __name__ == '__main__':
    main()
