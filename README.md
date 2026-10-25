# Рекомендательные системы на VK-LSVD

Фролов Никита Сергеевич

Датасет: [VK-LSVD](https://huggingface.co/datasets/deepvk/VK-LSVD),
срез `up0.001_ip0.001`, ревизия `084a18386e149a454dd6dbe0123478e75c6b09ce`.

Сравнение персональных top-10 рекомендаций видео: ItemKNN, iALS, EASE,
Factorization Machine с метапризнаками и iALS с непрерывным целевым сигналом.
Выборка содержит 3000 пользователей, 2000 видео и 1 483 828 взаимодействий.
Временное разбиение: недели 00–24 — train, 25 — validation, 26 — test.
Основная метрика — NDCG@10, дополнительные — Recall@10 и MAP@10.

## Запуск

Python 3.12+. Из корня проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name vk-lsvd --display-name "Python (VK-LSVD)"
python -m jupyterlab experiments.ipynb
```

Выберите ядро `Python (VK-LSVD)`.
В `experiments.ipynb` сохранены EDA, формулы моделей, пространства поиска,
таблицы, графики и выводы. По умолчанию `REPRODUCE=False`: ячейки используют
сохранённые результаты без повторного обучения. Для обучения и оценки
установите `REPRODUCE=True` и выполните notebook с начала.

Данные загружаются автоматически при повторном обучении. Отдельная загрузка:

```bash
python -m models.prepare_data
```

Команды повторения EDA и подбора параметров приведены в разделе 8 notebook.
Реализации моделей находятся в `models/`, общие функции — в `models/utils/`,
код экспериментов на validation — в `models/experiments/`.
Данные и промежуточные результаты создаются локально и исключены из Git.
