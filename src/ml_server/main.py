from fastapi import FastAPI

from dotenv import load_dotenv
from pathlib import Path

import multiprocessing as mp

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()

MAX_PROCESSORS = 4

# Счетчик занятых процессоров
busy_processors = mp.Value('i', 0)
