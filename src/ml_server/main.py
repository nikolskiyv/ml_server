from dotenv import load_dotenv
from fastapi import FastAPI

from src.ml_server import routes

app = FastAPI()

app.include_router(routes.router)


load_dotenv()
