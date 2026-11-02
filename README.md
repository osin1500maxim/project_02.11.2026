# project_02.11.2026 — оценка цены автомобиля по фото

Нейросеть (ConvNeXt-Tiny) распознаёт марку, модель, цвет и кузов по фото, а градиентный бустинг
оценивает цену по году выпуска и пробегу. Цена переводится в рубли по курсу ЦБ РФ, результат — PDF-отчёт и сайт.

## Состав
| Файл / папка | Что это |
|---|---|
| `train.py` | обучение: отбор данных DVM-CAR, подбор гиперпараметров (Optuna), нейросеть и модель цены |
| `predict.py` | оценка из командной строки: `python predict.py фото.jpg 2017 85000` → PDF |
| `site/` | сайт на FastAPI + HTML/CSS/JS (страницы «Оценка» и «О проекте»), Docker для сервера |

## Данные
Датасет [DVM-CAR](https://deepvisualmarketing.github.io/) (CC BY-NC) в репозиторий не входит.
Скачайте таблицы и архив фото и положите рядом с `train.py`:
- таблицы: https://figshare.com/articles/figure/DVM-CAR_Dataset/19586296/2
- фото спереди (730 МБ): https://figshare.com/ndownloader/files/34792480

## Запуск
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python train.py
python predict.py фото.jpg 2017 85000
```
Сайт: см. `site/README.md`.
