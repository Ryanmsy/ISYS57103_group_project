FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    curl \
    git \
    rustc \
    cargo \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY src/amazon_test_2500.xlsx /app/src/


CMD [ "python","src/sentiment_analysis_test.py" ]



