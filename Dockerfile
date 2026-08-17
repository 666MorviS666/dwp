# Панель — долгоживущий процесс: фоновый поток опрашивает Steam, копит
# историю матча и пишет лог. Контейнер такому подходит, а serverless нет
# (разбор — в DEPLOY.md).
FROM python:3.12-slim

# libgomp1 — единственная системная зависимость проекта. lightgbm собран с
# OpenMP, и без этой библиотеки падает не при обучении, а сразу на импорте,
# при загрузке .so. В slim-образе её нет.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DWP_HOME=/app

# Зависимости отдельным слоем и ДО кода: правка web.py не должна тянуть за
# собой пересборку колёс numpy и lightgbm (это минуты против секунд).
COPY dwp/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

# Проверка живости бьёт по самой странице, а не по /api/status: последний
# отвечает и тогда, когда модель не загрузилась, — то есть «здоров» стоял бы
# на мёртвой панели.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8765/', timeout=4)"

# 0.0.0.0 обязателен: 127.0.0.1 внутри сетевого пространства контейнера
# значит «слушать только себя», и снаружи панель недоступна.
CMD ["python", "-m", "dwp.web", "--host", "0.0.0.0", "--port", "8765"]
