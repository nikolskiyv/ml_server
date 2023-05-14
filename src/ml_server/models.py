from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, root_validator


class MLModelConfig(BaseModel):
    model: str
    params: Optional[dict]

    '''
    @validator('model')
    def model_name_validator(self, v):
        if not model_mapper.get(v):
            raise ValueError(f'Model {v} not supported by the server.')
        return v
    '''


class MLModelStatus(Enum):
    FITTING = 1  # Модель обучается
    FITTED = 2  # Модель обучена
    LOADED = 3  # Модель обучена и загружена
    UNLOADED = 4  # Модель выгружена
    REMOVED = 5  # Модель удалена (?!)


class FitBody(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: MLModelConfig

    @root_validator()
    def check_sizes(cls, values):
        if len(values['X']) != len(values['y']):
            raise ValueError("Mismatched sizes between X and y.")
        return values

    class Config:
        arbitrary_types_allowed = True


class PredictBody(BaseModel):
    X: List[List[float]]
    config: MLModelConfig

    class Config:
        arbitrary_types_allowed = True
