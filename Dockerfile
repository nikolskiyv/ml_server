FROM python:3-slim-buster

WORKDIR /app
RUN mkdir data

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY /src /app/src

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "src.ml_server.main:app", "--host=0.0.0.0", "--port=8000"]