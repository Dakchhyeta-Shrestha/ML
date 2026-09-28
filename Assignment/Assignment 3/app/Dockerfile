FROM python:3.11-slim
WORKDIR /project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model.py train.py ./
COPY app ./app
EXPOSE 8050
CMD ["python", "app/app.py"]
