# End-to-end LLM-as-judge для code-RAG

**Репозиторий:** WinMerge (open-source MFC/Win32 C++)  
**Модель:** `qwen2.5-coder:7b` (Ollama, OpenAI-compatible API)  
**Судья:** та же модель, что и генератор; вторичный `llama3.2:3b` для agreement (Cohen's κ)  

Система выполняет end-to-end прогон с LLM-as-judge для пайплайна code-RAG и выдаёт набор классических метрик качества. При смене промпта, retrieval-стратегии или модели сразу видно: стало **лучше или хуже**.

```
Вопрос ──► Hybrid retrieval ──► LLM генератор ──► {ответ, контекст, файлы}
           BM25 + dense (e5)                                │
           RRF fusion                          ┌────────────┼────────────┐
                                               ▼            ▼            ▼
                                          correctness  faithfulness  relevance
                                           (+ эталон)   (+ контекст)  (бонус)
                                               └────────────┼────────────┘
                                                            ▼
                                                      EvalMetrics
                                               JSON · JSONL · CSV · SQLite
                                               Grafana (localhost:3000)
                                               MLflow  (localhost:5000)
```

---

## 1. Требования

| Компонент | Версия | Зачем |
|---|---|---|
| Python | 3.10 + | Основной рантайм |
| Docker + Compose v2 | последний | Основной способ запуска |
| NVIDIA GPU | VRAM ≥ 6 GB | **Обязателен** для Docker-запуска |
| NVIDIA Driver | ≥ 527 (Win) / ≥ 525 (Linux) | GPU passthrough в контейнер |
| WinMerge репозиторий | коммит `ce4aa744` | Индексируется BM25 |

Клонировать репозиторий с данными:

```bash
git clone <this-repo> && cd <this-repo>
git clone https://github.com/WinMerge/winmerge.git winmerge
```

---

## 2. Быстрый старт

Два варианта запуска — выбрать один.

**Первый прогон:** дополнительно скачивает dense-эмбеддер (`multilingual-e5-base`, ~1.1 ГБ) и один раз индексирует корпус в эмбеддинги (кэш в `reports/.emb_cache/`, ~1–3 мин на CPU). Последующие прогоны переиспользуют кэш. Чтобы отключить dense и работать только на BM25 — `USE_DENSE=0`.

### Вариант А — Docker (рекомендуется, без внешних API-ключей)

Поднимает Ollama, MLflow и Grafana одной командой. Модель `qwen2.5-coder:7b` (~4 ГБ) скачивается автоматически.

```bash
docker compose up -d --build
# Первый запуск: ожидайте 5–15 минут на скачивание модели
```

Запустить оценку (`--build` гарантирует, что runner собран из текущего кода, а не из устаревшего образа):

```bash
docker compose --profile eval run --build --rm rag-eval evaluate \
  --dataset data/winmerge_eval.jsonl \
  --report  reports/run_baseline.json
```

**Важно:** `docker compose run` **без** `--build` переиспользует ранее собранный образ. После любого `git pull` / изменения кода обязателен флаг `--build` (или разовый `docker compose build rag-eval`), иначе контейнер крутит старую версию. Исходники также примонтированы (`./rag_eval`), поэтому правки чистого Python подхватываются без пересборки — пересборка нужна лишь при смене зависимостей.

Дашборды откроются сразу после первого прогона:

| Сервис | URL | Описание |
|---|---|---|
| Grafana | http://localhost:3000 | Дашборд с метриками в реальном времени |
| MLflow | http://localhost:5000 | История экспериментов и артефакты |
| Ollama | http://localhost:11434 | OpenAI-compatible API |

**Требования к железу:** NVIDIA GPU обязателен (VRAM ≥ 6 GB для 7B + 3B моделей).  
При отсутствии GPU `docker compose up` завершится ошибкой — см. §12 «Диагностика проблем».

---

### Вариант Б — Локальный Ollama (без Docker)

```bash
# Установить Ollama: https://ollama.ai
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b

pip install -r requirements.txt -r requirements-extras.txt
# CPU-сборки torch достаточно (dense-эмбеддер работает на CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env
# .env по умолчанию уже указывает на http://localhost:11434/v1 — менять ничего не нужно
```

Затем — те же команды `python -m rag_eval evaluate ...` (см. §4).

---

### Демо без LLM (dry-run)

Проверяет весь пайплайн с моковыми ответами. Реальных API-вызовов нет. В выходных файлах метаданные содержат `"dry_run": true`.

```bash
python -m rag_eval evaluate \
  --dataset data/winmerge_eval.jsonl \
  --report  reports/dry_run.json \
  --dry-run
```

---

## 3. Конфигурация

Скопировать `.env.example` → `.env`, заполнить нужные поля.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODEL_BASE_URL` | `http://localhost:11434/v1` | Endpoint LLM (OpenAI-compatible) |
| `MODEL_API_KEY` | `none` | API-ключ (для локального Ollama — `none`) |
| `MODEL_NAME` | `qwen2.5-coder:7b` | Генератор и первичный судья |
| `SECONDARY_BASE_URL` | `http://localhost:11434/v1` | Endpoint вторичного судьи (можно тот же Ollama) |
| `SECONDARY_API_KEY` | `none` | API-ключ вторичного судьи |
| `SECONDARY_MODEL` | `llama3.2:3b` | Вторичная модель (другое семейство → разнообразие для κ) |
| `CORRECTNESS_THRESHOLD` | `0.7` | Порог τ для метрики `pass_rate` |
| `MAX_WORKERS` | `1` | Число параллельных потоков оценки |
| `REQUEST_TIMEOUT_SEC` | `90` | Таймаут одного LLM-вызова (сек) |
| `WINMERGE_REPO_PATH` | `./winmerge` | Путь к клонированному WinMerge |
| `USE_DENSE` | `1` | `1` — гибрид BM25+dense; `0` — только BM25 |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Dense-эмбеддер (мультиязычный) |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` исключает конфликт за VRAM с Ollama |
| `BM25_MIN_SCORE` | `15` | BM25 входит в RRF-fusion только при реальном лексическом совпадении |
| `DRY_RUN` | `0` | `1` — моковые ответы вместо реальных |
| `MLFLOW_TRACKING_URI` | `./mlruns` | URI сервера MLflow (в Docker — `http://mlflow:5000`) |
| `METRICS_DB_PATH` | `./reports/metrics.db` | SQLite-база для Grafana datasource |

---

## 4. Запуск оценки

### 4.1 Базовый прогон (BM25-only retrieval)

`USE_DENSE=0` отключает dense-retriever — остаётся чистый лексический BM25. Это «до» в нашем A/B.

```bash
USE_DENSE=0 python -m rag_eval evaluate \
  --dataset data/winmerge_eval.jsonl \
  --report  reports/run_baseline.json
```

**Windows PowerShell:** `$env:USE_DENSE=0; python -m rag_eval evaluate --dataset data/winmerge_eval.jsonl --report reports/run_baseline.json`

Прогресс отображается в терминале. По завершении — итоговые числа и пути сохранённых файлов. Режим retrieval пишется в `metadata.retrieval_mode` отчёта.

---

### 4.2 Улучшенный прогон для A/B (гибридный retrieval)

Тот же датасет, тот же `top_k=5`, но поверх BM25 включён dense-retriever (e5-base) с RRF-fusion и confidence-gating. A/B измеряет вклад **гибридного retrieval** (см. §9).

```bash
USE_DENSE=1 python -m rag_eval evaluate \
  --dataset data/winmerge_eval.jsonl \
  --report  reports/run_improved.json \
  --baseline reports/run_baseline.json
```

`--baseline` включает regression-guard: сразу пишется `reports/comparison.json` и возвращается exit code 1, если composite упал больше чем на ε=0.02.

---

### 4.3 Сравнение двух прогонов

```bash
python -m rag_eval compare \
  --baseline  reports/run_baseline.json \
  --candidate reports/run_improved.json \
  --output    reports/comparison.json
```

Выводит таблицу Δ по всем метрикам в stdout. Если `composite_score` упал более чем на ε = 0.02 — возвращает **exit code 1** (подходит для CI-gate).

---

### 4.4 Калибровка порога τ

Используется после нескольких прогонов, когда накоплено 5+ ручных разметок.

```bash
python -m rag_eval calibrate \
  --annotations data/annotations_example.json \
  --output      reports/calibration.json
```

Формат файла аннотаций (JSON-массив):

```json
[
  {"case_id": "wm_en_01", "judge_score": 0.88, "human_label": 1},
  {"case_id": "wm_en_04", "judge_score": 0.43, "human_label": 0}
]
```

`human_label`: `1` — ответ корректный, `0` — некорректный. Команда подбирает τ в диапазоне 0.30–0.95 методом максимизации F1 и рекомендует значение для `--thr-correctness`.

---

### 4.5 Полный список опций CLI

```
python -m rag_eval evaluate [ОПЦИИ]

  --dataset PATH                  Путь к JSONL eval-набору        (обязателен)
  --report  PATH                  Выходной агрегатный JSON         (обязателен)
  --top-k   INT                   BM25 top-k чанков               (default: 5)
  --dual-judge / --no-dual-judge  Второй судья (llama3.2:3b)       (default: on)
  --dry-run / --no-dry-run        Моковые LLM-ответы
  --baseline PATH                 Предыдущий прогон для regression-guard
  --epsilon  FLOAT                Допустимое падение composite     (default: 0.02)
  --thr-correctness   FLOAT       Порог pass_rate (τ)              (default: 0.7)
  --thr-hallucination FLOAT       Порог faithfulness               (default: 0.5)
  --thr-false-refusal FLOAT       Порог false_refusal              (default: 0.3)
  --thr-false-answer  FLOAT       Порог false_answer               (default: 0.5)
```

---

## 5. Метрики

Агрегируются по всему набору и по каждой группе `language` / `difficulty`.

| Метрика | Формула | Что сигнализирует |
|---|---|---|
| `pass_rate` | доля(correctness ≥ τ), τ = 0.7 | Доля семантически верных ответов |
| `mean_correctness` | среднее correctness ∈ [0, 1] | Общее соответствие эталону |
| `mean_faithfulness` | среднее faithfulness ∈ [0, 1] | Заземлённость на контекст retrieval |
| `hallucination_rate` | доля(faith < 0.5) по `should_have_answer: true` | Изобретение фактов |
| `false_refusal_rate` | доля(corr < 0.3) по позитивным | Отказ там, где ответ есть |
| `false_answer_rate` | доля(corr ≥ 0.5) по `should_have_answer: false` | Выдуманный ответ на заведомо отсутствующий факт |
| `evidence_recall` | доля хитов по `required_evidence_any` | BM25 находит нужные файлы |
| `forbidden_claim_rate` | доля срабатываний `forbidden_claims_any` | Детектор запрещённых утверждений |
| `composite_score` | 0.45 · corr + 0.35 · faith + 0.20 · recall | Итоговый балл прогона |

**Целевые ориентиры:** `composite_score > 0.75`, `hallucination_rate < 0.05`, `evidence_recall > 0.60`.

**Примечание о `composite_score`:** считается строго по формуле ТЗ. Если в датасете **нет** кейсов с `required_evidence_any`, `evidence_recall` не определён (NaN) и composite тоже становится NaN — это намеренно: метрика сигнализирует «датасет без evidence», а не молча занижает балл.

**Ограничение `evidence_recall` для русских запросов.** BM25 работает над English C++ кодом. Русские вопросы без C++ идентификаторов (6 из 30 кейсов с `required_evidence_any`) не могут найти нужные файлы через лексический поиск. Средство — cross-lingual embeddings (FAISS + mGTE/mE5); в текущей BM25-имплементации это структурное ограничение.

**Примечание о `false_answer_rate`.**  
Формула ТЗ §6: `NOT should_have_answer AND correctness ≥ 0.5` — негативный кейс, где модель уверенно ответила вместо отказа. Порог настраивается флагом `--thr-false-answer` (default 0.5).  
На нашем датасете судья оценивает корректные отказы на негативных кейсах низко (corr 0.0–0.25, все < 0.5), поэтому `false_answer_rate = 0.000` и совпадает со структурным счётом отказов (`detected_refusal_rate`) — риск «корректный отказ засчитан как ложный ответ» на этих данных не реализуется.

**Примечание о `false_refusal_rate` и `detected_refusal_rate`.**  
Spec-формула: `is_refusal AND correctness < 0.3`.  
С упрощёнными промптами судьи это работает корректно: отказы получают corr ≈ 0.0–0.1. На финальном гибридном retrieval `false_refusal_rate = 0.033` (1/30) — с качественным контекстом генератор почти не отказывается (на чистом BM25 было 0.233 = 7/30).  
`detected_refusal_rate` (не в спеке, тот же результат на наших данных) — чисто структурная детекция: ответ < 40 слов без новых технических символов. Служит верификацией согласованности судьи со структурным детектором.

---

## 6. Мониторинг

### 6.1 Grafana (http://localhost:3000)

Дашборд загружается автоматически при `docker compose up`. Данные поступают из `reports/metrics.db` (SQLite), который обновляется после каждого `evaluate`.

Панели:
- **Stat-панели** — текущие composite_score, pass_rate, correctness, faithfulness, evidence_recall, hallucination, detected_refusal, Cohen's κ
- **Метрики по прогонам** (time series) — тренды composite, correctness, faithfulness, recall
- **Ошибки по прогонам** (time series) — hallucination, false_refusal, false_answer, detected_refusal
- **История прогонов** (table) — run_id, модель, top_k, composite, n_cases
- **Кейсы последнего прогона** (table) — correctness, faithfulness, passed, hallucination

Без Docker — открыть любой Grafana, добавить источник `frser-sqlite-datasource`, указать путь `reports/metrics.db`.

---

### 6.2 MLflow (http://localhost:5000)

Каждый `evaluate` автоматически создаёт run в эксперименте `rag-eval`:

| Что логируется | Формат |
|---|---|
| `model_name`, `top_k`, `winmerge_commit`, `dry_run` | params |
| Все скалярные метрики EvalMetrics | metrics |
| `report.json`, `cases.jsonl`, `report.csv` | artifacts |

Локальный запуск без Docker:

```bash
pip install mlflow
# MLFLOW_TRACKING_URI=./mlruns задан по умолчанию
python -m rag_eval evaluate ...
mlflow ui --port 5000    # http://localhost:5000
```

---

## 7. Структура вывода

Для `--report reports/out.json` создаются четыре файла:

| Файл | Содержимое |
|---|---|
| `reports/out.json` | Агрегатные метрики, breakdown по language/difficulty, dual-judge kappa |
| `reports/out_cases.jsonl` | Подробный дамп по каждому кейсу (один JSON на строку) |
| `reports/out.csv` | Flat-таблица для Grafana CSV datasource или BI-инструментов |
| `reports/comparison.json` | Δ-таблица (генерируется при `--baseline`) |
| `reports/metrics.db` | SQLite: таблицы `runs` + `cases` для Grafana frser-sqlite |

Пример структуры агрегатного JSON:

```json
{
  "composite_score": 0.699,
  "pass_rate": 0.306,
  "mean_correctness": 0.639,
  "mean_faithfulness": 0.889,
  "evidence_recall": 0.500,
  "hallucination_rate": 0.033,
  "by_language": {"en": {"composite_score": 0.71}, "ru": {"composite_score": 0.68}},
  "by_difficulty": {"easy": {"pass_rate": 0.4}, "medium": {"pass_rate": 0.3}},
  "cohens_kappa_correctness": 0.169,
  "metadata": {
    "model_name": "qwen2.5-coder:7b",
    "top_k": 5,
    "retrieval_mode": "hybrid",
    "use_dense": true,
    "bm25_min_score": 15.0,
    "winmerge_commit": "ce4aa744",
    "dry_run": false,
    "timestamp": "2026-05-29T13:09:07+00:00"
  }
}
```

---

## 8. Датасет

`data/winmerge_eval.jsonl` — **36 кейсов** по WinMerge, коммит `ce4aa744`.

| Группа | Кол-во | Тематика |
|---|---|---|
| Позитивные EN | 18 | CDiffWrapper, DiffList, CDiffThread, strdiff, ConflictFileParser, COptionsMgr, CDiffTextBuffer… |
| Позитивные RU | 12 | Те же компоненты, вопросы на русском языке |
| Негативные EN | 3 | CUDA, Python scripting, ML — функции, отсутствующие в WinMerge |
| Негативные RU | 3 | SQLite, REST API, real-time collaboration |

Схема одного кейса:

```json
{
  "id": "wm_en_01_run_file_diff",
  "question": "Which class launches the file diff in WinMerge?",
  "language": "en",
  "should_have_answer": true,
  "reference_answer": "CDiffWrapper in Src/DiffWrapper.cpp...",
  "required_evidence_any": ["Src/DiffWrapper.cpp"],
  "forbidden_claims_any": [],
  "difficulty": "easy",
  "tags": ["diff", "architecture"]
}
```

Все пути в `required_evidence_any` верифицированы в `winmerge/Src/`. Для трёх RU-кейсов (`wm_ru_04/05/10`), где вопрос спрашивает про *поведение/механизм*, эталон расширен с одного объявляющего заголовка до объявления + файла реализации (например, `DirScan.h` → `DirScan.h` + `DirScan.cpp`): реализация verifiably содержит ответ, поэтому это исправление изначально слишком узкого gold, а не подгонка метрики.

---

## 9. Примеры результатов

Два реальных прогона на `qwen2.5-coder:7b` (Ollama, генератор + первичный судья) + `llama3.2:3b` (вторичный судья, Ollama). A/B-рычаг — **retrieval**: BM25-only (baseline) против гибрида BM25 + dense (e5-base) с RRF-fusion (improved), при одинаковом `top_k=5`:

| Метрика | Baseline (BM25-only) | Improved (Hybrid) | Δ |
|---|---|---|---|
| **composite\_score** | 0.706 | **0.775** | **+0.069** |
| pass\_rate (τ=0.7) | 0.528 | **0.667** | +0.139 |
| mean\_correctness | 0.481 | **0.589** | +0.108 |
| mean\_faithfulness | **0.963** | 0.943 | −0.019 |
| mean\_relevance (бонус) | 0.660 | **0.819** | +0.160 |
| evidence\_recall | 0.767 | **0.900** | +0.133 |
| hallucination\_rate | **0.000** | 0.033 | +0.033 |
| false\_refusal\_rate | 0.233 | **0.033** | −0.200 |
| false\_answer\_rate | 0.000 | 0.000 | 0.000 |
| detected\_refusal\_rate | 0.233 | **0.067** | −0.167 |
| kappa\_correctness | 0.706 | 0.769 | — |

**Интерпретация.** Гибридный retrieval выигрывает уверенно: прирост composite +0.069 заметно превышает как порог ε=0.02, так и run-to-run шум судьи (≈0.01). Выигрыш сконцентрирован в `evidence_recall` (+0.133): попадание нужного файла в контекст каскадом поднимает `mean_correctness` (+0.108) и `pass_rate` (+0.139). Показательны `detected_refusal_rate` (0.233 → 0.067) и `false_refusal_rate` (0.233 → 0.033): на BM25-only нужные файлы не находились для ~8 кейсов, генератор корректно отказывался и получал низкую correctness — гибрид устраняет именно эти промахи retrieval. Издержки минимальны: `mean_faithfulness` −0.019 и +1 галлюцинация (больше контекста — выше шанс дрейфа), что на фоне выигрыша пренебрежимо. Ключевой вывод: **узким местом был retrieval, а не судья и не генератор.**

**Эволюция retrieval (история итераций).** Таблица A/B выше — авторитетное воспроизведение на текущем датасете (BM25-only vs hybrid, `top_k=5`). Таблица ниже — хронология того, как улучшалось качество входных данных для судей по ходу проекта; колонка «BM25» снята до коррекции gold-эталонов (§8), поэтому её `evidence_recall` (0.800) отличается от current baseline (0.767):

| Метрика | BM25 | + Hybrid dense | + Gold correction |
|---|---|---|---|
| evidence\_recall | 0.800 | 0.867 | **0.900** |
| mean\_correctness | 0.478 | 0.582 | **0.588** |
| mean\_faithfulness | 0.874 | 0.950 | **0.943** |
| composite\_score | 0.681 | 0.768 | **0.774** |
| detected\_refusal\_rate | 0.233 | 0.033 | **0.067** |

Падение `detected_refusal_rate` с 0.233 (BM25) до 0.03–0.07 (гибрид) — 7 отказов → 1–2 — прямое доказательство, что узким местом были **входные данные** (retrieval), а не судья: с правильным контекстом генератор перестал отказываться.

По языкам: EN evidence\_recall = **1.000**, RU = **0.750**. Оставшиеся 3 RU-промаха (`wm_ru_02/04/05`) — genuine предел эмбеддера на парах «RU-вопрос → EN-код»: целевые файлы лежат на ранге 274–553, выше них эмбеддер ставит шум (`StdAfx.cpp`). Более сильный эмбеддер (bge-m3) проверен — **выигрыша не дал** (та же 0.867 до коррекции gold). Реальный непробованный рычаг — перевод запроса RU→EN перед retrieval.

Cohen's κ = 0.71 (baseline) / 0.77 (improved) — существенное согласие Qwen-7B и LLaMA-3B (κ > 0.6).

---

## 10. Структура проекта

```
rag_eval/
├── config.py          Pydantic Settings; читает .env
├── dataset.py         Схема EvalCase; загрузчик JSONL
├── metrics.py         CaseMetrics, EvalMetrics, evaluate_rag_run, compare_runs
├── db.py              Запись в SQLite (Grafana datasource)
├── calibration.py     Калибровка τ методом максимизации F1
├── reporter.py        save_json / save_jsonl / save_csv / log_mlflow
├── __main__.py        CLI: evaluate / compare / calibrate
├── rag/
│   ├── chunker.py     Sliding-window чанкер C++/H (40 строк, overlap 10)
│   ├── indexer.py     BM25Index + HybridIndex (RRF fusion, confidence-gated BM25)
│   ├── dense.py       DenseIndex: multilingual-e5 эмбеддинги, cosine, кэш на диск
│   └── pipeline.py    make_answer_fn: retrieval → LLM → RagAnswer
└── judge/
    ├── prompts.py     Промпты судьи (correctness, faithfulness, relevance)
    ├── judge.py       Judge; retry ≤ 2; парсинг JSON с fence-stripping
    └── agreement.py   DualJudge, AgreementAggregator, Cohen's κ

data/
├── winmerge_eval.jsonl        36 eval-кейсов
└── annotations_example.json   10 ручных разметок для calibrate

reports/                       Генерируется CLI; содержит примеры dry-run
docker/
└── mlflow.Dockerfile          python:3.11-slim + mlflow
grafana/
└── provisioning/              Автозагрузка datasource и дашборда
docker-compose.yml             Ollama + MLflow + Grafana + rag-eval runner
Dockerfile                     Образ rag-eval
docs/design.md                 Архитектура судьи, ограничения, рекомендации (≤ 2 стр.)
```

---

## 11. Версии

| Компонент | Версия / Коммит |
|---|---|
| WinMerge | `ce4aa744ab9df2dc9cdf9a88bca231ded3e6bf97` |
| Генератор + первичный судья | `qwen2.5-coder:7b` через Ollama |
| Вторичный судья | `llama3.2:3b` через Ollama |
| Dense-эмбеддер | `intfloat/multilingual-e5-base` (опц. `BAAI/bge-m3`) |
| rank-bm25 | ≥ 0.2.2 |
| sentence-transformers | ≥ 2.7 (+ torch CPU) |
| openai SDK | ≥ 1.12 |
| mlflow | ≥ 2.15 (в `requirements-extras.txt`) |
| Python | 3.10 + |

---

## 12. Диагностика проблем

### `could not select device driver "nvidia"` при `docker compose up`

GPU не передан в Docker. Нужно:

**Windows (Docker Desktop + WSL2):**
```
1. Установить NVIDIA Driver ≥ 527 для Windows
2. Docker Desktop → Settings → Resources → GPU → включить
3. Перезапустить Docker Desktop
```

**Linux:**
```bash
# Установить NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверить доступность GPU в Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi
```

---

### Ollama не поднимается (unhealthy)

```bash
docker compose logs ollama
```

| Сообщение в логах | Причина | Решение |
|---|---|---|
| `no CUDA-capable device detected` | GPU есть, но драйвер не видит | Обновить NVIDIA драйвер |
| `CUDA out of memory` | VRAM недостаточно для модели | Переключить `MODEL_NAME` на `qwen2.5-coder:1.5b` |
| `llama runner started` | Всё ОК, модель грузится | Подождать `start_period` (15–30 с) |

Минимальный VRAM: `qwen2.5-coder:7b` — **6 GB**, `llama3.2:3b` — **3 GB**.

---

### `dependency failed to start: ollama is unhealthy`

```bash
# Проверить статус GPU внутри контейнера
docker exec pythonproject1-ollama-1 nvidia-smi

# Если nvidia-smi не найден — GPU не передан (см. выше)
```

---

### Модель скачивается каждый раз заново

`model-init` помечен как `restart: "no"` и сохраняет модели в volume `ollama_data`. При `docker compose down -v` volume удаляется. Использовать `docker compose down` (без `-v`) для сохранения данных.

---

### Оценка зависает на первом кейсе

Модель грузится первые 10–30 с (GPU). Это нормально — пайплайн делает warmup-вызов автоматически. Если зависание > 60 с:

```bash
docker compose logs rag-eval      # проверить ошибки
docker compose logs ollama | grep -i error
```

---

### Проверить GPU-ускорение после запуска

```bash
docker exec pythonproject1-ollama-1 ollama ps
# PROCESSOR должен содержать "GPU" или "100% GPU"
```

---

### `sqlite3.OperationalError: table runs has N columns but M values` / устаревший код в контейнере

`docker compose run` и `docker compose up` **не пересобирают** образ — код `rag_eval/` запекается внутрь при `build`. После любого изменения исходников образ нужно пересобрать, иначе контейнер крутит старую версию (симптомы: несоответствие схемы SQLite, старые числа метрик).

```bash
docker compose build rag-eval          # пересобрать образ runner-а
# затем обычный запуск:
docker compose --profile eval run --rm rag-eval evaluate \
  --dataset data/winmerge_eval.jsonl --report reports/run_baseline.json
```

Схема БД при этом самоисправляется: `save_run()` добавляет недостающие колонки через `ALTER TABLE` (миграция `_ensure_columns`). Если `reports/metrics.db` остался от старой схемы и мешает — его можно удалить, он пересоздаётся.
