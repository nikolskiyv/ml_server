from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, root_validator


class MLModelConfig(BaseModel):
    file_name: str
    model: str  # Название модели
    params: Optional[dict]  # Гиперпараметры модели

    '''
    @validator('model')
    def model_name_validator(self, v):
        if not model_mapper.get(v):
            raise ValueError(f'Model {v} not supported by the server.')
        return v
    '''


class FittingInfo(BaseModel):
    general: str  # Время обучение
    start: str  # Время начала обучения
    end: str  # Время конца обучения


class FitBody(BaseModel):
    X: List[List[float]]
    y: List[float]
    sync: str = 'No'
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
