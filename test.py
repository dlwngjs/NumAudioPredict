from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import numpy as np
import librosa
from sklearn.metrics import accuracy_score

trainSet = {
    0: ["path/f1/f10.wav", "path/f2/f20.wav", "path/f3/f30.wav", "path/f4/f40.wav", "path/m1/m10.wav", "path/m2/m20.wav", "path/m3/m30.wav"],
    1: ["path/f1/f11.wav", "path/f2/f21.wav", "path/f3/f31.wav", "path/f4/f41.wav", "path/m1/m11.wav", "path/m2/m21.wav", "path/m3/m31.wav"],
    2: ["path/f1/f12.wav", "path/f2/f22.wav", "path/f3/f32.wav", "path/f4/f42.wav", "path/m1/m12.wav", "path/m2/m22.wav", "path/m3/m32.wav"],
    3: ["path/f1/f13.wav", "path/f2/f23.wav", "path/f3/f33.wav", "path/f4/f43.wav", "path/m1/m13.wav", "path/m2/m23.wav", "path/m3/m33.wav"],
    4: ["path/f1/f14.wav", "path/f2/f24.wav", "path/f3/f34.wav", "path/f4/f44.wav", "path/m1/m14.wav", "path/m2/m24.wav", "path/m3/m34.wav"],
    5: ["path/f1/f15.wav", "path/f2/f25.wav", "path/f3/f35.wav", "path/f4/f45.wav", "path/m1/m15.wav", "path/m2/m25.wav", "path/m3/m35.wav"],
    6: ["path/f1/f16.wav", "path/f2/f26.wav", "path/f3/f36.wav", "path/f4/f46.wav", "path/m1/m16.wav", "path/m2/m26.wav", "path/m3/m36.wav"],
    7: ["path/f1/f17.wav", "path/f2/f27.wav", "path/f3/f37.wav", "path/f4/f47.wav", "path/m1/m17.wav", "path/m2/m27.wav", "path/m3/m37.wav"],
    8: ["path/f1/f18.wav", "path/f2/f28.wav", "path/f3/f38.wav", "path/f4/f48.wav", "path/m1/m18.wav", "path/m2/m28.wav", "path/m3/m38.wav"],
    9: ["path/f1/f19.wav", "path/f2/f29.wav", "path/f3/f39.wav", "path/f4/f49.wav", "path/m1/m19.wav", "path/m2/m29.wav", "path/m3/m39.wav"]
}

testSet = {
    0: ["path/m4/m40.wav"],
    1: ["path/m4/m41.wav"],
    2: ["path/m4/m42.wav"],
    3: ["path/m4/m43.wav"],
    4: ["path/m4/m44.wav"],
    5: ["path/m4/m45.wav"],
    6: ["path/m4/m46.wav"],
    7: ["path/m4/m47.wav"],
    8: ["path/m4/m48.wav"],
    9: ["path/m4/m49.wav"]
}

# Custom wrapper for GMM
class GMMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=1, covariance_type='diag', max_iter=100):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.gmm = None

    def fit(self, X, y=None):
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=42
        )
        self.gmm.fit(X)
        return self

    def score(self, X, y=None):
        # Return the log-likelihood score
        return self.gmm.score(X)


def get_mfcc(file):
    y, sr = librosa.load(file, sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, n_fft=512, n_mels=40, hop_length=128)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.hstack((mfcc.T, mfcc_delta.T, mfcc_delta2.T))

    return features


def train_models_with_grid_search(trainSet):
    best_models = {}
    param_grid = {
        'n_components': range(1, 10, 2),
        'covariance_type': ['full', 'diag', 'tied', 'spherical'],
        'max_iter': [100, 200, 300, 400]
    }

    for label, files in trainSet.items():
        print(f"Training model for label {label}...")
        mfcc_features = np.concatenate([get_mfcc(file) for file in files], axis=0)

        # Use GridSearchCV with GMMWrapper
        gmm_grid = GridSearchCV(
            estimator=GMMWrapper(),
            param_grid=param_grid,
            scoring=None,  # Use estimator's score method
            cv=5  # 3-fold cross-validation
        )
        gmm_grid.fit(mfcc_features)
        best_models[label] = gmm_grid.best_estimator_
        print(f"Best parameters for label {label}: {gmm_grid.best_params_}")
    return best_models


def predict(gmm_models, test_file):
    mfcc_features = get_mfcc(test_file)
    scores = {label: gmm.score(mfcc_features) for label, gmm in gmm_models.items()}
    # print(f"{test_file} Score :")
    # for label, score in scores.items():
    #     print(f"  Model {label}: {score:.2f}")
    return max(scores, key=scores.get)


def main():
    gmm_models = train_models_with_grid_search(trainSet)

    y_true = []
    y_pred = []
    for label, files in testSet.items():
        for file in files:
            y_true.append(label)
            y_pred.append(predict(gmm_models, file))

    accuracy = accuracy_score(y_true, y_pred)
    print(f"y_true : {y_true}\ny_pred : {y_pred}")
    print(f"Acc : {accuracy}")


if __name__ == '__main__':
    main()
