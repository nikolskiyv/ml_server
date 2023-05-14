from fastapi import FastAPI
from ml_server import routes

app = FastAPI()

app.include_router(routes.router)

