import multiprocessing
from multiprocessing import Process

from fastapi import FastAPI, HTTPException

from ml_server.main import busy_processors, MAX_PROCESSORS
from ml_server.ml_models.base import MLModelAPI, remove_all
from ml_server.models import FitBody, PredictBody

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/fit")
async def fit_model(body: FitBody):
    if busy_processors.value >= MAX_PROCESSORS:
        raise HTTPException(status_code=503, detail="Все процессы заняты. Попробуйте позже.")

    with busy_processors.get_lock():
        busy_processors.value += 1

    process = Process(target=MLModelAPI.fit, args=(body.X, body.y, body.config, busy_processors))
    process.start()

    return {"msg": f"Запущен новый процесс для {body.config.model}. "
                   f"Занято процессов: {busy_processors.value}/{MAX_PROCESSORS}"}


@app.post("/predict")
async def predict_model(body: PredictBody):
    prediction = MLModelAPI.predict(body.X, body.config)
    return {'prediction': list(prediction)}


@app.post("/remove_all")
async def remove_all_models():
    await remove_all()
    return {"message": "all models removed"}

'''
@app.post("/load")
async def load_model(config: MLModelConfig):
    MLModelAPI.load(config)
    return {"message": "model loaded"}


@app.post("/unload")
async def unload_model(config: MLModelConfig):
    MLModelAPI.unload(config)
    return {"message": "model unloaded"}


@app.post("/remove")
async def remove_model(config: MLModelConfig):
    MLModelAPI.remove(config)
    return {"message": "model removed"}
'''

