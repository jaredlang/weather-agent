FROM python:3.9 

EXPOSE 8501

CMD mkdir -p /app 

WORKDIR /app

COPY requirements.txt ./requirements.txt 

RUN pip install --no-cache-dir -r requirements.txt 

COPY . . 

CMD [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableWebsocketCompression=false"]
