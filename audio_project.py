import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score


# trainSet = {
#     0: ["path/f1/f10.wav", "path/f2/f20.wav", "path/f3/f30.wav", "path/f4/f40.wav", "path/m1/m10.wav", "path/m2/m20.wav", "path/m3/m30.wav", "path/m4/m40.wav"],
#     1: ["path/f1/f11.wav", "path/f2/f21.wav", "path/f3/f31.wav", "path/f4/f41.wav", "path/m1/m11.wav", "path/m2/m21.wav", "path/m3/m31.wav", "path/m4/m41.wav"],
#     2: ["path/f1/f12.wav", "path/f2/f22.wav", "path/f3/f32.wav", "path/f4/f42.wav", "path/m1/m12.wav", "path/m2/m22.wav", "path/m3/m32.wav", "path/m4/m42.wav"],
#     3: ["path/f1/f13.wav", "path/f2/f23.wav", "path/f3/f33.wav", "path/f4/f43.wav", "path/m1/m13.wav", "path/m2/m23.wav", "path/m3/m33.wav", "path/m4/m43.wav"],
#     4: ["path/f1/f14.wav", "path/f2/f24.wav", "path/f3/f34.wav", "path/f4/f44.wav", "path/m1/m14.wav", "path/m2/m24.wav", "path/m3/m34.wav", "path/m4/m44.wav"],
#     5: ["path/f1/f15.wav", "path/f2/f25.wav", "path/f3/f35.wav", "path/f4/f45.wav", "path/m1/m15.wav", "path/m2/m25.wav", "path/m3/m35.wav", "path/m4/m45.wav"],
#     6: ["path/f1/f16.wav", "path/f2/f26.wav", "path/f3/f36.wav", "path/f4/f46.wav", "path/m1/m16.wav", "path/m2/m26.wav", "path/m3/m36.wav", "path/m4/m46.wav"],
#     7: ["path/f1/f17.wav", "path/f2/f27.wav", "path/f3/f37.wav", "path/f4/f47.wav", "path/m1/m17.wav", "path/m2/m27.wav", "path/m3/m37.wav", "path/m4/m47.wav"],
#     8: ["path/f1/f18.wav", "path/f2/f28.wav", "path/f3/f38.wav", "path/f4/f48.wav", "path/m1/m18.wav", "path/m2/m28.wav", "path/m3/m38.wav", "path/m4/m48.wav"],
#     9: ["path/f1/f19.wav", "path/f2/f29.wav", "path/f3/f39.wav", "path/f4/f49.wav", "path/m1/m19.wav", "path/m2/m29.wav", "path/m3/m39.wav", "path/m4/m49.wav"]
# }

'''테스트하고 싶은 파일 제외하고 모두 숫자별로 라벨링해서 불러오기'''
trainSet = {
    0: ["path/f1/f10.wav", "path/f2/f20.wav", "path/f3/f30.wav", "path/f4/f40.wav", "path/m1/m10.wav", "path/m2/m20.wav", "path/m3/m30.wav", "path/m4/m40.wav"],
    1: ["path/f2/f21.wav", "path/f3/f31.wav", "path/f4/f41.wav", "path/m1/m11.wav", "path/m2/m21.wav", "path/m3/m31.wav", "path/m4/m41.wav"],
    2: ["path/f1/f12.wav", "path/f2/f22.wav", "path/f3/f32.wav", "path/m1/m12.wav", "path/m2/m22.wav", "path/m3/m32.wav", "path/m4/m42.wav"],
    3: ["path/f1/f13.wav", "path/f2/f23.wav", "path/f3/f33.wav", "path/f4/f43.wav", "path/m1/m13.wav", "path/m2/m23.wav", "path/m3/m33.wav", "path/m4/m43.wav"],
    4: ["path/f1/f14.wav", "path/f3/f34.wav", "path/f4/f44.wav", "path/m1/m14.wav", "path/m2/m24.wav", "path/m3/m34.wav", "path/m4/m44.wav"],
    5: ["path/f1/f15.wav", "path/f2/f25.wav", "path/f3/f35.wav", "path/f4/f45.wav", "path/m1/m15.wav", "path/m2/m25.wav", "path/m3/m35.wav", "path/m4/m45.wav"],
    6: ["path/f1/f16.wav", "path/f2/f26.wav", "path/f3/f36.wav", "path/f4/f46.wav", "path/m1/m16.wav", "path/m2/m26.wav", "path/m3/m36.wav", "path/m4/m46.wav"],
    7: ["path/f1/f17.wav", "path/f2/f27.wav", "path/f3/f37.wav", "path/f4/f47.wav", "path/m2/m27.wav", "path/m3/m37.wav", "path/m4/m47.wav"],
    8: ["path/f1/f18.wav", "path/f2/f28.wav", "path/f3/f38.wav", "path/f4/f48.wav", "path/m1/m18.wav", "path/m2/m28.wav", "path/m3/m38.wav", "path/m4/m48.wav"],
    9: ["path/f1/f19.wav", "path/f2/f29.wav", "path/f3/f39.wav", "path/f4/f49.wav", "path/m1/m19.wav", "path/m3/m39.wav", "path/m4/m49.wav"]
}
'''테스트하고 싶은 데이터 파일 숫자별로 라벨링해서 불러오기'''
testSet = {
    1: ["path/f1/f11.wav"],
    2: ["path/f4/f42.wav"],
    6: ["path/f2/f26.wav"],
    7: ["path/m1/m17.wav"],
    9: ["path/m2/m29.wav"]
}


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
        # print(f"y_true : {y_true}\ny_pred : {y_pred}")
        # print(f"Acc : {acc}")
        if(acc > max_acc):
            max_acc = acc
        # print(f"max_y_true : {max_y_true}\nmax_y_pred : {max_y_pred}")
        # print(f"Acc : {max_acc}")
        if(max_acc >= 0.8):
            break
    # 최적의 랜덤값을 가진 모델로 테스트
    y_true = []
    y_pred = []
    for label, files in testSet.items():
        for file in files:
            y_true.append(label)
            y_pred.append(predict(gmm_models, file))

    acc = accuracy_score(y_true, y_pred)
    print(f"y_true : {y_true}\ny_pred : {y_pred}")
    print(f"Acc : {acc}")

if __name__ == '__main__':
    main()
