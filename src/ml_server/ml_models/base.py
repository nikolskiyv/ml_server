import os
from typing import List

import joblib

from ml_server.main import MAX_PROCESSORS
from ml_server.models import MLModelConfig


from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


model_mapper = {
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'LogisticRegression': LogisticRegression,
    'SVM': SVC,
    'KNeighborsClassifier': KNeighborsClassifier,
}


class MLModelAPI:
    @classmethod
    def fit(cls, X, y, config: MLModelConfig, busy_processors):
        """
        Обучение модели и сохранение на диск по указанному имени.
        """
        model = model_mapper.get(config.model)
        model_path = os.path.join('test', f"{config.model}.joblib")
        model = model(**config.params) if config.params else model()

        model.fit(X, y)
        cls.unload(model, model_path)

        with busy_processors.get_lock():
            busy_processors.value -= 1
        print(f'Процесс {os.getpid()} освобожден. '
              f'Занято процессов: {busy_processors.value}/{MAX_PROCESSORS}')

    @classmethod
    def predict(cls, X, config) -> List[float]:
        """
        Предсказание с помощью обученной и загруженной модели по её имени.
        """
        model = cls.load(config)
        predictions = model.predict(X)
        return predictions

    @staticmethod
    def load(config):
        """
        Загрузка обученной модели по её имени в режим инференса.
        """
        model_path = os.path.join('test', f"{config.model}.joblib")

        if not os.path.exists(model_path):
            return "Модель не загружена."

        return joblib.load(model_path)

    @staticmethod
    def unload(model, model_path) -> None:
        """
        Выгрузка загруженной модели по её имени.
        """
        joblib.dump(model, model_path)

    @staticmethod
    def remove(config) -> None:
        """
        Удаление обученной модели с диска по её имени.
        """
        model_path = os.path.join('test', f"{config.model}.joblib")
        os.remove(model_path)


async def remove_all():
    model_dir = os.getenv('MODEL_PATH')

    for filename in os.listdir(model_dir):
        if filename.endswith(".joblib"):
            file_path = os.path.join(model_dir, filename)
            os.remove(file_path)
