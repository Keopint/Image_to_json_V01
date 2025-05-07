FROM python:3.9-slim

FROM redis:alpine as redis
FROM python:3.9-slim

COPY --from=redis /usr/local/bin/redis-cli /usr/local/bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1 \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app/__init__.py
ENV FLASK_ENV=development
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

RUN chmod +x entrypoint.sh

EXPOSE 5000


CMD ["./entrypoint.sh"]