FROM python:3.10-slim

ENV MODEL_DIR=/models \
    MODEL_FILENAME=model.bin \
    LOG_LEVEL=INFO

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/models"]

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]