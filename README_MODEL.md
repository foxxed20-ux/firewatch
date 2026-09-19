# FireWatch: конкурсные модели AF и BS

Этот пакет реализует локальную сегментацию активного горения AF (классы 0/1) и гари BS (классы 0/1/2/3). Сервис разрабатывается отдельно и использует `competition.service_bridge`; HTTP не требуется для конкурсного инференса.

## Данные и протокол

Источник: официальный открытый набор организаторов, `https://disk.yandex.ru/d/-rpmevTflbXZQg`. Используется 420 AF и 224 BS train-чипа. В `artifacts/split.json` зафиксированы 515 train и 129 validation: AF 336/84, BS 179/45. У AF нет заполненного fire_event_id, поэтому разбиение изолирует геосетки; у BS изолированы события и геосетки. Подробности и ограничения — `artifacts/data_inspection.json`.

На вход модели не поступают координаты, даты, ID событий, статистика целевой маски или test metadata. Validation используется для выбора модели и постобработки; её результаты не являются независимым закрытым test Score. Test не используется для обучения или подбора параметров.

AF имеет 8 полос VIIRS и 5 AUX; BS — по 10 полос Sentinel-2 до/после, по 2 полосы Sentinel-1 и 3 AUX. Фактический BS AUX не содержит aspect, несмотря на расширенное описание в постановке. Reader проверяет имена, типы, размеры и совпадение геосеток. Модель получает 23 AF-признака или 55 BS-признаков, включая временные разности, индексы и локальный контекст. Нормализация CNN оценивается только на train; LightGBM принимает физически масштабированные признаки.

Облака и маски наблюдения остаются признаками и не исключают пиксели из конкурсного результата. `NaN/Inf` входа заменяются нулём детерминированно; доступный флаг валидности сохраняется отдельным признаком. Исключение из loss/validation разрешено только для реальной целевой NoData-метки 255; в проверенном наборе таких пикселей не обнаружено.

## Установка и инференс

Python 3.10+; рекомендуем отдельное окружение:

```bash
python -m venv .venv
# Активируйте окружение средствами вашей ОС.
pip install -r requirements-model.txt
python inference.py --data-dir /path/to/test --output submission.csv
```

Для CPU-сервера можно установить CPU-сборку PyTorch; при полностью LightGBM bundle PyTorch для инференса не требуется. `model_bundle/manifest.json` должен находиться рядом с `inference.py`; альтернативный путь задаётся `--model-dir` или `MODEL_ROOT`.

CLI читает настоящий `sample_submission.csv`, сохраняет точный порядок и множество строк, проверяет классы и RLE round-trip, пишет UTF-8 CSV с quoted empty strings. При ошибке завершение ненулевое; частичный результат не подменяет итоговый CSV. Все зависимости и веса размещаются заранее, инференс работает без сети.

```python
from competition.service_bridge import describe_models, predict_chip

status = describe_models(model_dir="model_bundle")
result = predict_chip("data/demo", "AF_tr_000126", model_dir="model_bundle")
mask = result["class_map"]
```

Результат: `class_map` uint8 HW, информационная `observation_valid` bool HW, `score` и `provenance`. AF score — HW, BS score — CHW. Score не заявлен как калиброванная вероятность. Bundle проверяется по SHA-256; несовместимые или неполные веса отклоняются. Реальная готовность сервиса также требует загрузки и контрольного предсказания, а не только проверки хешей.

## Воспроизведение обучения

Распакуйте официальный train отдельно от test:

```bash
python -m competition.prepare --data-dir data/train --output prepared --split artifacts/split.json
python train.py --records prepared/records.json --output runs/cnn-v1 --task af --epochs 80 --batch-size 8 --time-limit-min 14 --workers 1
python train.py --records prepared/records.json --output runs/cnn-v1 --task bs --epochs 120 --batch-size 2 --time-limit-min 43 --workers 1
python -m competition.tree --records prepared/records.json --output runs/tree-v1 --task af --max-pixels 800000 --val-pixels 200000 --rounds 700 --num-threads 2
python -m competition.tree --records prepared/records.json --output runs/tree-v1 --task bs --max-pixels 1000000 --val-pixels 250000 --rounds 700 --num-threads 2
python -m competition.evaluate --records prepared/records.json --output runs/eval-v1 --cnn-root runs/cnn-v1 --tree-root runs/tree-v1
```

Бюджет CNN мягкий, проверяется между эпохами. Повторение в другой среде может завершить другое число эпох; для сравнения используйте зафиксированную историю, версии и число эпох выбранной модели. Псевдослучайное начальное состояние — 42. CNN обучается с нуля; внешние веса и внешняя разметка не используются.

Метрика: `Score = 0.35*F1_af + 0.35*IoU_burn + 0.30*mIoU_sev`. TP/FP/FN суммируются по всем чипам; при пустых предсказании и эталоне компонент равен 1. BS-классы взаимоисключающие. Выбор порога AF и коэффициентов BS выполняется только по validation.

Фактические итоговые результаты и сведения о поставке будут добавлены после завершения обучения и полного прогона.
