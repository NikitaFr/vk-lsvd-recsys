"""baselines experiment on train/validation."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parents[2])
from IPython.display import display

from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display


def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def locate_root(start):
    for p in (start, *start.parents):
        if (p / 'data/work/config.json').is_file() and (p / 'data/work/interactions.parquet').is_file():
            return p
    raise FileNotFoundError('Открой notebook в исходном проекте vk_lsvd_eda_starter (или подкаталоге)')


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_ids(ids):
    a = np.asarray(ids)
    require(a.ndim == 1 and a.dtype.kind in 'iu', 'ID должны быть целочисленными')
    require(not (a < 0).any(), 'Неожиданные отрицательные ID')
    return hashlib.sha256(np.sort(a.astype(np.uint64)).tobytes()).hexdigest()

ROOT = locate_root(Path.cwd().resolve())
OUT = ROOT / 'results/baselines/validation'
ITEMKNN_DIR = ROOT / 'results/item_knn/validation'
FILES = {
    'train': ROOT / 'data/work/interactions.parquet',
    'users': ROOT / 'data/work/users_metadata.parquet',
    'items': ROOT / 'data/work/items_metadata.parquet',
    'validation': ROOT / 'data/source/subsamples/up0.001_ip0.001/validation/week_25.parquet',
}
EXPECTED_SHA = {
    'train': '0c3a20a9729e30aeb650b28379d34663d3d2ec914bfe0ee3ca0a17d0904790a0',
    'users': 'c0769c126e96499364f902362902690774005947d9fe23e097e43718d26d5cb2',
    'items': '3140e6749978a817e552a510c91f0c402eb93a639a096dc81c0bc99526b27d71',
    'validation': 'd654b40a7f7e2086054c6131767967f84289fafd2364f566b984cc7d408917f5',
}
EXPECTED_ID_SHA = {
    'users': 'ce51fd99e116ae5884714fcaf2244b6b99c4d5268ed961426ea86b8277c7780b',
    'items': 'e69b0688d7d2ff4a68375a9bc711026d58899b565ce93c4c4c94692c29b59e76',
}
EXPECTED_EVAL_MODULES = {
    '__init__.py': 'e509fa0eaad88e80bdab56c2fc205660068753d5386db837a574c66d41b2a04c',
    '_validation.py': '6c84b47d675919f6fa6e17526acdcb61736c973ad9d92fd2a3c7ab72f54bf405',
    'bootstrap.py': '9dba42b565ee3cbc10ff5c5773644fd12a95e6c64a6a193a84a7b92776bca221',
    'metrics.py': '14091e44dddbf13e8eb0e49290ac6f4bb15ceeb7d5e98c2d9b2760a23692b938',
    'pipeline.py': '27f4bb63e439c9e9ab4bba4f99b7776bcc78f9c7500566e38657de624b79178f',
    'ranking.py': '5c224821db66c3ebd5367bf764d36a975b5147db3dd249e959af3fc7240159e2',
    'relevance.py': 'ff8f64f467b73b7d62fc797d26e4dc7da606bea2ee1b169e02c7791b1bbfd7e1',
}
for name, f in FILES.items():
    require(f.is_file(), f'Не найден исходный файл {f}')
    require(sha_file(f) == EXPECTED_SHA[name], f'Исходный {name} НЕ совпадает с EDA')
for file, sha in EXPECTED_EVAL_MODULES.items():
    f = ROOT / 'models/utils/evaluation' / file
    require(f.is_file() and sha_file(f) == sha, f'Неверная версия evaluator: {file}')
require((ITEMKNN_DIR / 'selection_and_manifest.json').is_file(),
        'Сначала положи в проект ранее полученные results/item_knn/validation/')
require((ITEMKNN_DIR / 'winner_binary_normalized_per_user.csv').is_file(),
        'Не найден сохранённый per_user лучшего Binary ItemKNN')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.utils.evaluation import (label_relevance, evaluate_scores,
                            bootstrap_mean, bootstrap_paired_difference)
print('PASS: файлы train/val, evaluator и предыдущее исследование ItemKNN найдены и сверены')
print('ROOT:', ROOT)
print('Python:', sys.executable)


TRAIN_COLUMNS = ['user_id','item_id','week','timespent','like','bookmark','dislike']
VAL_COLUMNS = ['user_id','item_id','timespent','like','bookmark','dislike']
train_raw = pd.read_parquet(FILES['train'], columns=TRAIN_COLUMNS)
users_meta = pd.read_parquet(FILES['users'], columns=['user_id'])
items_meta = pd.read_parquet(FILES['items'], columns=['item_id','duration'])
val_source = pd.read_parquet(FILES['validation'], columns=VAL_COLUMNS)
require(len(train_raw) == 1_454_216 and set(train_raw.week.unique()) == set(range(25)),
        'Не совпало содержимое train')
require(len(users_meta) == 3000 and len(items_meta) == 2000, 'Неверный каталог')
require(len(val_source) == 519_839, 'Не та официальная validation-неделя')
require(not users_meta.user_id.duplicated().any() and not items_meta.item_id.duplicated().any(),
        'Дубли в метаданных')
USER_IDS = np.sort(users_meta.user_id.to_numpy(dtype=np.uint64))
ITEM_IDS = np.sort(items_meta.item_id.to_numpy(dtype=np.uint64))
require(sha_ids(USER_IDS) == EXPECTED_ID_SHA['users'], 'User IDs не совпали с EDA')
require(sha_ids(ITEM_IDS) == EXPECTED_ID_SHA['items'], 'Item IDs не совпали с EDA')
require(set(train_raw.user_id) == set(USER_IDS) and set(train_raw.item_id) == set(ITEM_IDS),
        'Train содержит другие ID')
val_raw = val_source.loc[
    val_source.user_id.isin(USER_IDS) & val_source.item_id.isin(ITEM_IDS)
].copy()
val_raw['week'] = np.int16(25)  # Исходный файл week_25.parquet подтверждён SHA-256.
require(len(val_raw) == 16_158 and val_raw.user_id.nunique() == 2819 and
        val_raw.item_id.nunique() == 839, 'Другая validation после фиксации ID')
require(not train_raw.duplicated(['user_id','item_id']).any() and
        not val_raw.duplicated(['user_id','item_id']).any(), 'Дубликаты user-item')
require(not pd.MultiIndex.from_frame(val_raw[['user_id','item_id']]).isin(
    pd.MultiIndex.from_frame(train_raw[['user_id','item_id']])).any(),
    'Пересечение пар train и validation')

def with_relevance(frame):
    return label_relevance(frame.merge(items_meta, on='item_id', how='left', validate='many_to_one'))

train = with_relevance(train_raw)
validation = with_relevance(val_raw)
require(train.rel.value_counts().sort_index().to_dict() == {0:922110,1:148820,2:383286},
        'Распределение train rel не совпало с EDA')
require(validation.rel.value_counts().sort_index().to_dict() == {0:11352,1:1462,2:3344},
        'Распределение validation rel не совпало с EDA')
HISTORY = train[['user_id','item_id','week','rel']].copy()
HOLDOUT = validation[['user_id','item_id','week','rel']].copy()
require(len(HISTORY) == 1_454_216 and len(HOLDOUT) == 16_158, 'Неверная длина истории/holdout')
print('PASS: исходная выборка 3000×2000, train 0–24, validation 25, test не читался')


SEEDS = (42,43,44,45,46)
METRICS = ('ndcg_at_10','recall_at_10','ap_at_10')
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

positive = HISTORY.loc[HISTORY.rel >= 1]
counts = positive.groupby('item_id').size().reindex(ITEM_IDS, fill_value=0)
require(int(counts.sum()) == 532_106, 'Popularity использовала неправильные train-positives')
require(np.array_equal(counts.index.to_numpy(dtype=np.uint64), ITEM_IDS),
        'Порядок popularity item_id отличается от каталога')
POP_COUNTS = counts.to_numpy(dtype=np.float64)
# Широковещательная матрица: модель не обучается, score(i)=число train-positive для item i.
POP_SCORES = np.broadcast_to(POP_COUNTS, (len(USER_IDS), len(ITEM_IDS)))

def run_eval(scores):
    result = evaluate_scores(scores=scores, user_ids=USER_IDS, item_ids=ITEM_IDS,
                             history=HISTORY, holdout=HOLDOUT, k=10,
                             history_start_week=0, history_end_week=24, holdout_week=25)
    require(result.summary['n_users'] == 3000 and result.summary['n_eligible'] == 2082 and
            result.summary['n_history'] == 1_454_216 and result.summary['n_holdout'] == 16_158 and
            result.summary['holdout_week_checked'], 'Evaluator применил другую постановку')
    return result

def macro_row(name, result, seed=None):
    s = result.summary
    return {'model':name, 'seed':seed,
            'eligible_ndcg_at_10':s['eligible']['ndcg_at_10'],
            'eligible_recall_at_10':s['eligible']['recall_at_10'],
            'eligible_map_at_10':s['eligible']['map_at_10'],
            'all_users_ndcg_at_10':s['all_users']['ndcg_at_10'],
            'all_users_recall_at_10':s['all_users']['recall_at_10'],
            'all_users_map_at_10':s['all_users']['map_at_10']}

popularity_result = run_eval(POP_SCORES)
random_rows = []
random_primary = None
for seed in SEEDS:
    # Случайные уникальные с большой вероятностью float64 scores для каждого user-item.
    scores_random = np.random.default_rng(seed).random((len(USER_IDS), len(ITEM_IDS)))
    ev = run_eval(scores_random)
    random_rows.append(macro_row('Random', ev, seed))
    if seed == SEEDS[0]:
        random_primary = ev
    del scores_random
random_runs = pd.DataFrame(random_rows)
require(len(random_runs) == 5 and random_primary is not None, 'Неверный набор random-seeds')

for name, result in [('Random seed=42',random_primary),('Positive Popularity',popularity_result)]:
    rec = result.recommendations
    per = result.per_user
    require(len(rec) == 30_000 and len(per) == 3_000, f'{name}: неверное число результатов')
    require(rec.groupby('user_id').size().eq(10).all(), f'{name}: не Top-10')
    require(not rec.duplicated(['user_id','item_id']).any(), f'{name}: повтор видео в Top-10')
    require(not pd.MultiIndex.from_frame(rec[['user_id','item_id']]).isin(
        pd.MultiIndex.from_frame(HISTORY[['user_id','item_id']])).any(), f'{name}: утечка просмотренного')
    require(int(per.eligible.sum()) == 2082 and int(per.n_relevant.sum()) == 4806,
            f'{name}: другое множество validation positives')

# Точное математическое ожидание Random Recall@10 при равномерной выдаче допустимых видео.
seen = HISTORY.groupby('user_id').item_id.nunique().reindex(USER_IDS,fill_value=0)
candidates = len(ITEM_IDS) - seen.to_numpy()
elig = popularity_result.per_user.eligible.to_numpy()
random_recall_expectation = float(np.mean(10 / candidates[elig]))
require(np.isfinite(random_recall_expectation) and 0 < random_recall_expectation < 1,
        'Некорректное ожидаемое случайное Recall')
print('PASS: Random×5 и Positive Popularity рассчитаны на полном каталоге и фиксированной истории')
print('Ожидаемый Random Recall@10 (аналитически):',round(random_recall_expectation,7))
display(pd.DataFrame([macro_row('Random seed=42',random_primary),
                      macro_row('Positive Popularity',popularity_result)]))
print('Разброс Random по заранее фиксированным seeds:')
display(random_runs[['seed','eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']])


prior = json.loads((ITEMKNN_DIR / 'selection_and_manifest.json').read_text(encoding='utf-8'))
require(prior['status'] == 'PASS_VALIDATION_ONLY' and not prior['test_data_read'],
        'Предыдущий эксперимент должен быть validation-only')
require(prior['user_ids_sha256'] == EXPECTED_ID_SHA['users'] and
        prior['item_ids_sha256'] == EXPECTED_ID_SHA['items'],
        'Сохранённый ItemKNN использовал другие ID')
require(prior['input_file_sha256'] == EXPECTED_SHA,
        'Сохранённый ItemKNN использовал другие входные Parquet')
chosen = prior['selection']['best_binary']
require(chosen['signal'] == 'binary' and chosen['aggregation'] == 'normalized' and
        chosen['k_neighbors'] == 640, 'Сохранённые лучшие параметры отличаются от согласованных')
knn_user = pd.read_csv(ITEMKNN_DIR / 'winner_binary_normalized_per_user.csv',
                       dtype={'user_id':np.uint64})
require(np.array_equal(knn_user.user_id.to_numpy(dtype=np.uint64), USER_IDS),
        'ItemKNN per_user содержит не те ID или порядок')
ref_user = popularity_result.per_user
require(np.array_equal(knn_user.eligible.to_numpy(), ref_user.eligible.to_numpy()) and
        np.array_equal(knn_user.n_relevant.to_numpy(), ref_user.n_relevant.to_numpy()),
        'Не совпадает eligible-маска или число positives')
for metric, key in [('ndcg_at_10','eligible_ndcg_at_10'),
                    ('recall_at_10','eligible_recall_at_10'),
                    ('ap_at_10','eligible_map_at_10')]:
    require(np.isclose(knn_user.loc[knn_user.eligible, metric].mean(),chosen[key],atol=1e-12),
            f'ItemKNN {metric} не совпал с сохранённым отчётом')
knn_summary = {'eligible':{'ndcg_at_10':chosen['eligible_ndcg_at_10'],
                           'recall_at_10':chosen['eligible_recall_at_10'],
                           'map_at_10':chosen['eligible_map_at_10']},
               'all_users':{'ndcg_at_10':chosen['all_users_ndcg_at_10'],
                            'recall_at_10':chosen['all_users_recall_at_10'],
                            'map_at_10':chosen['all_users_map_at_10']}}
rows = [macro_row('Random seed=42',random_primary),
        macro_row('Positive Popularity',popularity_result),
        {'model':'ItemKNN binary normalized k=640','seed':None,
         **{f'{group}_{metric}':value for group, metrics in knn_summary.items()
            for metric,value in metrics.items()}}]
comparison = pd.DataFrame(rows)
print('Сравнение трёх фиксированных моделей на validation:')
display(comparison[['model','eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10']])

ci_rows=[]
for name, df in [('Random seed=42',random_primary.per_user),
                 ('Positive Popularity',popularity_result.per_user),
                 ('ItemKNN binary normalized k=640',knn_user)]:
    for metric in METRICS:
        for group in ('eligible','all_users'):
            ci=bootstrap_mean(df,metric=metric,group=group,
                              n_bootstrap=N_BOOTSTRAP,ci=0.95,seed=BOOTSTRAP_SEED)
            ci_rows.append({'model':name,'metric':'map_at_10' if metric=='ap_at_10' else metric,
                            'group':group,**ci})
ci_table=pd.DataFrame(ci_rows)

pair_rows=[]
for label,a,b in [
    ('ItemKNN − Popularity',knn_user,popularity_result.per_user),
    ('ItemKNN − Random seed=42',knn_user,random_primary.per_user),
    ('Popularity − Random seed=42',popularity_result.per_user,random_primary.per_user),
]:
    for metric in METRICS:
        for group in ('eligible','all_users'):
            d=bootstrap_paired_difference(a,b,metric=metric,group=group,
                                          n_bootstrap=N_BOOTSTRAP,ci=0.95,seed=BOOTSTRAP_SEED)
            pair_rows.append({'comparison':label,'metric':'map_at_10' if metric=='ap_at_10' else metric,
                              'group':group,**d})
paired_table=pd.DataFrame(pair_rows)
print('Парные разности Eligible NDCG@10 (CI 95%):')
display(paired_table.loc[(paired_table.metric=='ndcg_at_10') &
                         (paired_table.group=='eligible'),
                         ['comparison','estimate','ci_low','ci_high']])


chart_data = comparison.melt(
    id_vars=['model'],
    value_vars=['eligible_ndcg_at_10','eligible_recall_at_10','eligible_map_at_10'],
    var_name='metric',value_name='value')
chart_data['metric'] = chart_data.metric.map({
    'eligible_ndcg_at_10':'NDCG@10','eligible_recall_at_10':'Recall@10',
    'eligible_map_at_10':'MAP@10'})
fig=px.bar(chart_data,x='model',y='value',color='model',facet_col='metric',
           title='VK-LSVD · Validation: Random / Popularity / ItemKNN',
           category_orders={'model':['Random seed=42','Positive Popularity',
                                     'ItemKNN binary normalized k=640'],
                            'metric':['NDCG@10','Recall@10','MAP@10']})
fig.update_yaxes(matches=None,showticklabels=True)
fig.update_layout(showlegend=False,margin=dict(t=75,b=120),height=470)
fig.update_xaxes(tickangle=25)
fig.show()

OUT.mkdir(parents=True,exist_ok=True)
comparison.to_csv(OUT/'comparison_metrics.csv',index=False)
random_runs.to_csv(OUT/'random_five_seeds.csv',index=False)
ci_table.to_csv(OUT/'bootstrap_metric_intervals.csv',index=False)
paired_table.to_csv(OUT/'bootstrap_paired_differences.csv',index=False)
popularity_result.per_user.to_csv(OUT/'popularity_per_user.csv',index=False)
popularity_result.recommendations.to_csv(OUT/'popularity_recommendations.csv',index=False)
random_primary.per_user.to_csv(OUT/'random_seed42_per_user.csv',index=False)
random_primary.recommendations.to_csv(OUT/'random_seed42_recommendations.csv',index=False)
fig.write_html(str(OUT/'baseline_comparison.html'),include_plotlyjs=True,full_html=True)

report={
    'status':'PASS_BASELINES_VALIDATION_ONLY',
    'dataset':'VK-LSVD', 'n_users':3000,'n_items':2000,
    'train_weeks':'0–24','validation_week':25,'test_data_read':False,
    'n_train':int(len(HISTORY)), 'n_validation':int(len(HOLDOUT)),
    'n_eligible':2082,'n_validation_positive':4806,
    'id_sha256':EXPECTED_ID_SHA,'input_sha256':EXPECTED_SHA,
    'random_seeds':list(SEEDS),'representative_random_seed':SEEDS[0],
    'random_expectation_recall_at_10':random_recall_expectation,
    'popularity_definition':'number of train-only interactions with rel >= 1 per item',
    'itemknn_comparison':'existing selected Binary Normalized ItemKNN k=640',
    'eligible_ndcg_at_10':{r['model']:float(r['eligible_ndcg_at_10'])
                            for r in comparison.to_dict('records')},
    'bootstrap_repeats':N_BOOTSTRAP,'bootstrap_seed':BOOTSTRAP_SEED,
    'warning':'Bootstrap CIs on same validation used for selecting ItemKNN parameters; descriptive, not held-out test inference.',
}
(OUT/'baseline_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('PASS: сохранены результаты Random и Popularity, сопоставление с ItemKNN')
print('Папка:', OUT)


RUN_K_SENSITIVITY = True   # Поставь False, чтобы пропустить только этот дополнительный эксперимент.

if RUN_K_SENSITIVITY:
    expected_model_sha='06ab92db064f1c0f793d6d136beef9289bcedcbd6676e095a34707887a728bc4'
    require(sha_file(ROOT/'models/item_knn.py') == expected_model_sha,
            'Версия ItemKNN отличается от проверенной ранее')
    from models import ItemKNN
    binary_knn = ItemKNN(signal='binary').fit(HISTORY,USER_IDS,ITEM_IDS,
                                              history_start_week=0,history_end_week=24)
    knn_640 = run_eval(binary_knn.score(k=640,aggregation='normalized'))
    knn_80 = run_eval(binary_knn.score(k=80,aggregation='normalized'))
    for metric in METRICS:
        require(np.allclose(knn_640.per_user[metric].to_numpy(),
                            knn_user[metric].to_numpy(),atol=1e-12,rtol=0,equal_nan=True),
                f'Повторный ItemKNN k=640 не воспроизводит сохранённые {metric}')
    delta_rows=[]
    for metric in METRICS:
        for group in ('eligible','all_users'):
            d=bootstrap_paired_difference(knn_640.per_user,knn_80.per_user,
                                          metric=metric,group=group,n_bootstrap=N_BOOTSTRAP,
                                          ci=0.95,seed=BOOTSTRAP_SEED)
            delta_rows.append({'comparison':'k=640 − k=80',
                               'metric':'map_at_10' if metric=='ap_at_10' else metric,
                               'group':group,**d})
    k_sensitivity=pd.DataFrame(delta_rows)
    k_sensitivity.to_csv(OUT/'itemknn_k640_vs_k80_paired.csv',index=False)
    knn_80.per_user.to_csv(OUT/'itemknn_k80_per_user.csv',index=False)
    report['k_sensitivity_ran']=True
    report['k_sensitivity_verified_k640_matches_saved']=True
    (OUT/'baseline_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS: k=640 полностью воспроизводится, k=80 сравнен парным bootstrap')
    display(k_sensitivity.loc[(k_sensitivity.metric=='ndcg_at_10') &
                               (k_sensitivity.group=='eligible'),
                               ['comparison','estimate','ci_low','ci_high']])
else:
    print('SKIP: k-sensitivity; baseline Random и Popularity уже сохранены')

print('Результаты сохранены.')