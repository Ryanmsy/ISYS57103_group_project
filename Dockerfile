FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY src/amazon_test_2500.xlsx /app/src/


CMD [ "python","src/sentiment_analysis_test.py" ]



