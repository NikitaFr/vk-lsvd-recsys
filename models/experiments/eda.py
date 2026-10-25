"""eda experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
from IPython.display import display, Markdown
import pandas as pd

from models.utils.eda import run_eda

ROOT = Path.cwd()
assert (ROOT / "data/work/config.json").exists(), "Открой Jupyter из корня проекта (где data/work/config.json)."
print("Корень проекта:", ROOT)
print("Рабочая выборка:", (ROOT / "data/work/interactions.parquet").exists())


tables, checks, figures = run_eda(ROOT)

print("Число взаимодействий:", f"{checks['events']:,}")
print("Пользователей:", f"{checks['unique_users']:,}")
print("Видео:", f"{checks['unique_items']:,}")
print("Плотность:", f"{checks['matrix_density']:.2%}")
print("Все количественные ограничения ДЗ выполнены:", checks['assignment_size_ok'])
print("Всего интерактивных графиков:", sum(len(v) for v in figures.values()))
print("Таблицы:", ROOT / "results/eda/distributions")


display(pd.DataFrame([checks]).T.rename(columns={0: "Значение"}))
display(tables["missingness"].sort_values("fraction", ascending=False).head(25))
for title, fig in figures["01_quality"]:
    display(fig)


display(tables["user_activity"].n_interactions.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_frame("Событий на пользователя"))
display(tables["item_popularity"].n_interactions.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_frame("Событий на видео"))
for title, fig in figures["02_activity"]:
    display(fig)


display(tables["by_duration"])
display(tables["watch_ratio_reactions"][["ratio_bin", "n"]])
for title, fig in figures["03_watch"]:
    display(fig)


display(tables["reactions"])
display(tables["reaction_conditional"])
for title, fig in figures["04_reactions"]:
    display(fig)


display(tables["user_summary"].describe(include="all").T)
for title, fig in figures["05_users"]:
    display(fig)


display(tables["contexts"].sort_values("n", ascending=False).head(20))
for title, fig in figures["06_metadata"]:
    display(fig)


display(tables["weekly"])
for title, fig in figures["07_time"]:
    display(fig)
