# Cripl EMO — эмо-стек Крипл (Affect Engine)

Локальная (офлайн) подсистема эмоций для проекта **«Крипл»**. Принимает текст, обновляет внутреннее аффективное состояние персонажа и выдаёт **snapshot** (структурированный снимок состояния) для GUI/логов/интеграции.

> Репозиторий содержит **готовый рабочий прототип**. Никаких сетевых запросов подсистема не делает: модель эмоций загружается **только из локальной папки**.

---

## Содержимое репозитория

Ключевые файлы:

- `kripl_affect.py` — ядро эмо-стека (**KriplAffect**): состояние, обновление, `snapshot()`
- `emo_backend_goemotions.py` — офлайн backend анализа текста (локальная Transformers-модель): вероятности меток + `pad_target`
- `engine.py` — фасад **AffectEngine** (единый интерфейс для GUI/тестов, выбор backend)
- `gui_affect.py` — отладочный GUI (Tkinter)
- `emo_config.json` — конфигурация (пути к модели + параметры)
- `appraisal_text.py` — утилита **TextAppraisal** (дельта-PAD и метрики по тексту без изменения состояния)
- `affect_core.py` — упрощённый движок (simple backend)
- `dual_affect.py` — альтернативный “dual” движок (если присутствует)


---

## Требования

- Python **3.10+**
- Для ML-backend: `torch`, `transformers`
- Для GUI: `tkinter` (обычно уже есть в стандартном Python на Windows)

Установка зависимостей (пример):
```bash
pip install torch transformers
```

---

## Быстрый старт (GUI)

**Важно:** запускайте GUI **из корня проекта**, чтобы относительные пути к конфигу работали корректно.

```bash
python gui_affect.py
```

GUI позволяет:
- ввести текст,
- сделать шаг обновления,
- посмотреть mood/burst/eff, метки и параметры,
- увидеть snapshot.

---

## Конфигурация (emo_config.json)

Файл `emo_config.json` задаёт:
- путь к локальной модели (секция `paths`, например `local_model_dir`)
- устройство `cpu`/`cuda` (секция `backend`)
- параметры динамики и ограничений (времена, клипы, гистерезис меток и т.д.)

### Локальная модель (офлайн)
Backend ожидает **локальную папку модели** в формате HuggingFace (например: `config.json`, токенизатор, веса `.bin/.safetensors`).

---

## Использование из кода

### Вариант A — напрямую через KriplAffect
```python
from kripl_affect import KriplAffect

emo = KriplAffect("emo_config.json")

emo.step("Привет! Как дела?")
print(emo.snapshot())

# учёт self-сигнала (ответ Крипл) — если используется в вашей интеграции
emo.update_from_self((0.1, 0.2, -0.1))
print(emo.snapshot())
```

### Вариант B — через фасад AffectEngine (рекомендуется для интеграции)
```python
from engine import load_engine

eng = load_engine("emo_config.json", backend="kripl")  # kripl | simple | dual

eng.step_text("Сообщение пользователя")
snap = eng.snapshot()
print(snap)

# если нужен «дрейф/затухание» между сообщениями:
eng.idle(dt=0.25)
```

### TextAppraisal (без изменения состояния)
```python
from appraisal_text import TextAppraisal

app = TextAppraisal()
out = app.analyze("Мне грустно и пусто.")
print(out["deltas"], out["metrics"])
```

---

## Snapshot (что возвращается)

`snapshot()` возвращает JSON-подобный словарь. Минимальная ожидаемая структура:

- `USER` — данные последнего пользовательского шага (текст, метка, вероятности, PAD)
- `KRIPL` — текущее состояние слоёв (mood/burst/eff), метки, marks и тайминг

Точный состав полей см. в `EMO_DOC.tex` (раздел **API и snapshot**).

---

## Частые проблемы

### `FileNotFoundError: Config not found: ...`
Путь к конфигу считается **от текущей рабочей директории (cwd)**.

Решения:
- запускайте `python gui_affect.py` из корня проекта
- или передавайте **абсолютный путь** к `emo_config.json`

### Ошибка загрузки модели `transformers`
Проверьте:
- что `paths.local_model_dir` указывает на папку с моделью
- что внутри есть `config.json`, tokenizer-файлы и веса

### CUDA недоступна
Поставьте `backend.device = "cpu"` в `emo_config.json` или установите корректный `torch` под вашу CUDA.

---

## Лицензия
Добавьте файл `LICENSE` под вашу задачу (обычно MIT).

---

## Контакты / вклад
Если вы меняете публичный API — обновляйте `EMO_DOC.tex` (раздел API) и README.
