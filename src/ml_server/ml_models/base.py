import base64
import glob
import gzip
import os
import time
from datetime import datetime
from typing import List

import joblib

from ml_server.models import MLModelConfig, FittingInfo

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ml_server.utils import MAX_PROCESSORS, logger

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
        logger.info(
            f"\t{datetime.now()} | Обработка запроса на обучение модели {config.model} | <PID:{os.getpid()}>"
        )
        model = model_mapper.get(config.model)
        model = model(**config.params) if config.params else model()

        logger.info(
            f"\t{datetime.now()} | Модель {model} ставится на обучение | <PID:{os.getpid()}>"
        )

        start_time = datetime.now()
        model.fit(X, y)
        end_time = datetime.now()

        logger.info(
            f"\t{datetime.now()} | Модель {model} обучена | <PID:{os.getpid()}>"
        )

        model_path = os.path.join('data', f"{config.file_name}.{start_time.isoformat()}_{end_time.isoformat()}.joblib")

        cls.unload(model, model_path)
        logger.info(
            f"\t{datetime.now()} | Модель {model} сохранена | <PID:{os.getpid()}>"
        )

        with busy_processors.get_lock():
            busy_processors.value -= 1
        logger.info(f'\t{datetime.now()} | Процесс <PID:{os.getpid()}> освобожден.'
                    f'Занято процессов: {busy_processors.value}/{MAX_PROCESSORS}')

    @classmethod
    def predict(cls, X, config) -> (List[float], None):
        """
        Предсказание с помощью обученной и загруженной модели по её имени.
        """
        model = cls.load(config)
        predictions = model.predict(X)
        return predictions, None

    @staticmethod
    def load(config):
        """
        Загрузка обученной модели по её имени в режим инференса.
        """
        model_path = os.path.join('data', f"{config.model}.joblib")

        pattern = f'data/{config.model}.*'

        matching_directories = glob.glob(pattern)
        print(matching_directories)

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
        model_path = os.path.join('data', f"{config.model}.joblib")
        os.remove(model_path)

    @classmethod
    def model_status(cls, model_name):
        directory = 'data'
        pattern = f"{model_name}*"  # Шаблон для поиска текстовых файлов

        # Используем glob.glob() для поиска файлов по шаблону
        file_list = glob.glob(f"{directory}/{pattern}")

        file_path = file_list[0]

        file_path = file_path[len(directory)+len(model_name)+2:-len('.joblib')]

        start, end = file_path.split('_')

        format_string = "%Y-%m-%dT%H:%M:%S.%f"
        fitting_start = datetime.strptime(start, format_string)
        fitting_end = datetime.strptime(end, format_string)

        fitting_time = fitting_end-fitting_start

        return FittingInfo(start=fitting_start.isoformat(),
                           end=fitting_end.isoformat(),
                           general=str(fitting_time))


async def remove_all():
    model_dir = os.getenv('MODEL_PATH')

    for filename in os.listdir(model_dir):
        if filename.endswith(".joblib"):
            file_path = os.path.join(model_dir, filename)
            os.remove(file_path)
