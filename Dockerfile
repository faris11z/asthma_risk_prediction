FROM python:3.11-slim

WORKDIR /app

COPY myapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY myapp/ .

EXPOSE 7860

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]
