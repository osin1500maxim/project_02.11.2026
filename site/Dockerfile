# Образ сайта для сервера без видеокарты
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# PyTorch для процессора (в 4 раза меньше версии с CUDA)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# скачиваем модель удаления фона заранее, чтобы первый посетитель не ждал
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY app.py core.py ./
COPY static ./static
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
