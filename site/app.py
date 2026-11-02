"""
БЭКЕНД САЙТА (FastAPI)

Запуск на своём компьютере:   uvicorn app:app --reload
Потом откройте в браузере:    http://127.0.0.1:8000

Как устроено:
  GET  /                 страница «Оценка»          (static/index.html)
  GET  /about            страница «О проекте»       (static/about.html)
  GET  /api/info         точность модели и список моделей авто (для страниц)
  POST /api/estimate     фото + год + пробег  ->  JSON с ценой и аналитикой
  GET  /api/report/{id}  PDF-отчёт по оценке
  GET  /api/health       проверка, что сервер жив (нужна для Docker)
"""
import base64
import datetime as dt
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import core

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
MAX_UPLOAD = 10 * 1024 * 1024          # 10 МБ — больше фото не принимаем
REPORT_TTL = 3600                      # PDF хранятся в памяти 1 час

models: core.Models | None = None
reports: dict[str, tuple[float, str, bytes]] = {}          # id -> (время, имя файла, PDF)
busy = threading.Semaphore(int(os.environ.get("MAX_PARALLEL", "2")))   # не больше N оценок одновременно


@asynccontextmanager
async def lifespan(_app):
    """Загружаем модели один раз при старте сервера."""
    global models
    models = core.Models()
    yield


app = FastAPI(title="Оценка автомобиля по фото", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------------ страницы
@app.get("/", include_in_schema=False)
def page_index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/about", include_in_schema=False)
def page_about():
    return FileResponse(os.path.join(STATIC, "about.html"))


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    return {"ok": models is not None}


@app.get("/api/info")
def info():
    q = models.quality
    metrics_txt = os.path.join(core.MODEL_DIR, "metrics.txt")
    return {
        "quality": q,
        "models": models.models_list,
        "n_models": len(models.models_list),
        "metrics_text": open(metrics_txt, encoding="utf-8").read() if os.path.exists(metrics_txt) else None,
        "has_curves": os.path.exists(os.path.join(core.MODEL_DIR, "training_curves.png")),
        "rate": dict(zip(("value", "date", "source"), core.gbp_rate())),
    }


@app.get("/api/curves.png", include_in_schema=False)
def curves():
    path = os.path.join(core.MODEL_DIR, "training_curves.png")
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/api/estimate")
def estimate(photo: UploadFile = File(...), year: int = Form(...), km: float = Form(...),
             model: str | None = Form(None)):
    # 1. проверяем ввод
    today = dt.date.today()
    if not 1990 <= year <= today.year:
        raise HTTPException(422, f"Год выпуска должен быть от 1990 до {today.year}")
    if not 0 <= km <= 1_500_000:
        raise HTTPException(422, "Пробег должен быть от 0 до 1 500 000 км")
    if model and model not in models.models_list:
        raise HTTPException(422, "Такой модели нет в списке")
    data = photo.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Фото больше 10 МБ")
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(422, "Не удалось открыть фото. Нужен JPG или PNG.")

    # 2. считаем (тяжёлая часть — не больше MAX_PARALLEL одновременно)
    if not busy.acquire(timeout=60):
        raise HTTPException(503, "Сервер занят, попробуйте через минуту")
    try:
        r = models.estimate(data, year, km, model or None)
        pdf = core.build_pdf(r, models.quality)
    finally:
        busy.release()

    # 3. сохраняем PDF и отдаём результат
    _cleanup()
    rid = uuid.uuid4().hex
    safe = "".join(c if c.isalnum() else "_" for c in r["model"])
    reports[rid] = (time.time(), f"otchet_{safe}_{year}.pdf", pdf)
    buf = io.BytesIO()
    r["_img"].save(buf, format="JPEG", quality=88)
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out["photo"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    out["report_url"] = f"/api/report/{rid}"
    return out


@app.get("/api/report/{rid}")
def report(rid: str):
    if rid not in reports:
        raise HTTPException(404, "Отчёт устарел — сделайте оценку ещё раз")
    _, name, pdf = reports[rid]
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _cleanup():
    now = time.time()
    for k in [k for k, (t, _, _) in reports.items() if now - t > REPORT_TTL]:
        reports.pop(k, None)
