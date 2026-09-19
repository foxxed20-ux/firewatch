# FireWatch: технический отчёт по моделям

**Статус:** технический отчёт, 19 сентября 2026 года.
**Стадия:** обучение и сравнение LightGBM, CNN и XGBoost завершены. Выбран и развёрнут V3: ансамбли 25 % CNN / 75 % LightGBM для AF и BS. Полный test CLI, независимая проверка CSV, CPU replay всех 129 validation-чипов и восемь сервисных проверок пройдены. Проверенный V2 сохранён для отката. Документ не содержит оценку private test.

## 1. Задача и протокол

Решение состоит из двух попиксельных модулей:

- **AF, active fire:** бинарная маска природного горения на VIIRS-чипах 256 × 256, GSD 375 м;
- **BS, burned severity:** маска 0/1/2/3 на Sentinel-1/2-чипах 512 × 512, GSD 20 м; 0 — фон, 1–3 — степени поражения.

Официальная метрика агрегирует пиксели всех чипов до вычисления показателей:

\[
Score=0.35\,F1_{AF}+0.35\,IoU_{burn}+0.30\,mIoU_{severity}.
\]

В `IoU_burn` классы BS 1–3 объединены в бинарную гарь. `mIoU_severity` — среднее IoU классов 1, 2 и 3. Если класс отсутствует и в эталоне, и в предсказании, компонент равен 1. Реализация находится в [`competition/metrics.py`](../competition/metrics.py).

Основания: «Постановка кейса — Мониторинг природных пожаров, КосмоХакатон 2026», стр. 9–16, и «Критерии оценки — Мониторинг природных пожаров, КосмоХакатон 2026», стр. 2–6. Все числа ниже относятся только к открытому train и нашему validation split. Test не использовался для обучения, выбора признаков, порогов или моделей.

## 2. Данные и проверенный контракт

Набор получен по [ссылке организаторов](https://disk.yandex.ru/d/-rpmevTflbXZQg). Фактическая схема зафиксирована в [`artifacts/train_schema.json`](../artifacts/train_schema.json), аудит — в [`artifacts/data_inspection.json`](../artifacts/data_inspection.json).

Неизменность полученных архивов зафиксирована в [`artifacts/data_source_lock.json`](../artifacts/data_source_lock.json): train — 2 305 976 320 байт, SHA-256 `9cc2d532312dd91d24beb8c6abcfdddac0cff6ca095501be28d81f751a8f159a`; test — 920 483 840 байт, SHA-256 `0431f8ae3af3a266a99074f87ca4a4caf5b6ea2e3645048a905ccac10c51674f`. Lock связывает эти архивы с хешами schema, inspection и split. Время генерации lock отмечено отдельно от filesystem mtime; точное время приобретения данных не утверждается.

| Модуль | Чипы | Вход | Цель |
|---|---:|---|---|
| AF | 420 | VIIRS I1–I5, solar/sensor zenith, valid; landcover, DEM, t2m, rh2m, wind speed | `active_fire`, 0/1 |
| BS | 224 | Sentinel-2 pre/post: B2–B8A, B11, B12, SCL; Sentinel-1 pre/post: VV/VH; DEM, slope, landcover | `severity`, 0/1/2/3 |

Reader проверяет число, порядок и имена полос, dtype, размер, CRS и affine transform. В данных наблюдались UTM-зоны EPSG:32637 и EPSG:32638.

Фактический BS AUX содержит три полосы: `dem`, `slope`, `landcover`. Упомянутого в расширенном описании `aspect` в выданных файлах нет, поэтому модель его не ожидает. Поле `landcover_top` в metadata пусто; тип покрова берётся из AUX-растра.

### 2.1. EDA → решение

**AF крайне несбалансирован.** В 296 из 420 чипов есть огонь, 124 чипа отрицательные. Всего 9 725 положительных пикселей из 27 525 120, около **0,0353 %**. Поэтому пиксели выбираются стратифицированно внутри train-чипов, sample weights восстанавливают исходный prior, а порог выбирается по полному global micro-F1 на validation.

**BS также несбалансирован.** В масках 5 909 001 пикселей гари; доли severity 1/2/3 равны **39,99 / 37,10 / 22,91 %**. Суммарная площадь по metadata — **236 360,04 га**. Поэтому применяются стратифицированная выборка, коррекция prior и validation-подбор множителей классов.

**Облачность не исключается из оценки.** По актуальному BS metadata медиана `cloud_frac` около **1,07 %**, 90-й процентиль — **25,33 %**, максимум — **48,48 %**. Минимальный `valid_frac` равен 45,97 % для AF и 51,52 % для BS. SCL и `valid` выступают признаками качества наблюдения; предсказание формируется для каждого конкурсного пикселя.

**Одного dNBR недостаточно.** Уборка и вспашка, сухая растительность, вода, облака и тени могут имитировать спектральное изменение. Поэтому к индексам добавлены исходные pre/post-каналы, их разности, SCL, тип покрова, рельеф и SAR.

**SAR даёт независимый сигнал.** Sentinel-1 не зависит от облачности и освещения. В модель входят VV/VH до и после события, их разности и локальное среднее изменения VH.

## 3. Split и защита от утечки

Зафиксированное разбиение — [`artifacts/split.json`](../artifacts/split.json), идентификатор `firewatch-split-v1`, SHA-256 `29a434ee0b2aff7b31cef766cd78bd452fd0241ad5bad3b8647b84ccb39a8f42`. Этот mapping является каноническим split для повторения экспериментов; отдельный генератор разбиения не поставляется. Повторное применение файла воспроизводимо, создание нового случайного split не эквивалентно этому эксперименту.

| Модуль | Train | Validation | Группировка | Пересечение |
|---|---:|---:|---|---:|
| AF | 336 | 84 | точная геосетка `(EPSG, bounds)` | 0 общих геосеток |
| BS | 179 | 45 | `fire_event_id` и точная геосетка | 0 общих событий и геосеток |
| Всего | 515 | 129 | — | — |

В AF 70 уникальных геосеток; 58 повторяются, максимальная кратность — 31. `fire_event_id` пуст. Связывание одновременно по геосетке и времени объединяет все 420 AF-чипов в один компонент. Поэтому жёстко изолирована география; даты могут встречаться с обеих сторон split. Validation проверяет перенос на новые сетки, но не является строгим spatial-temporal holdout.

В BS все 224 `fire_event_id` и 224 геосетки уникальны. Split стратифицирован по году, EPSG и квартилю `burn_area_ha`. Поля `n_fire_px`, `burn_area_ha`, `sev1_px`, `sev2_px`, `sev3_px` использованы только для split и диагностики, но не являются признаками.

Posthoc-проверка прямоугольников train/validation внутри одного EPSG не нашла ни точных дубликатов, ни пересечений положительной площади: AF — 53 train и 17 validation геосеток; BS — 179 и 45. Касания рёбер/углов не считаются пересечением. Сопоставление между разными EPSG не выполнялось. Воспроизводимый агрегированный отчёт без координат и ID — [`artifacts/split_overlap_audit.json`](../artifacts/split_overlap_audit.json), команда `ops.audit_split_overlap`.

Test metadata, координаты и даты не использовались; обратное геокодирование test не выполнялось; готовые пожарные продукты по test не применялись. Проверка утечки выполнена по идентификаторам и bounds, но не включает попиксельное хеширование всех снимков.

## 4. Предобработка и признаки

Контракт `official-named-v1` реализован в [`competition/data.py`](../competition/data.py).

### 4.1. AF: 23 признака

Вход включает I1–I5, углы, `valid`, landcover, DEM, t2m, rh2m, wind speed; температурные контрасты `I4-I5`, `I4-t2m`, `I5-t2m`; нормированные отношения отражения; локальные средние и контрасты теплового сигнала в окнах 3 и 11 пикселей. I4/I5 и t2m центрируются относительно 273,15 K, углы и AUX масштабируются физически осмысленными константами.

### 4.2. BS: 55 признаков

Вход включает Sentinel-2 и Sentinel-1 до/после, DEM, slope, landcover; спектральные и SAR-разности; NBR, NDVI, NBR2, NDWI до и после и их изменения; локальные средние dNBR/dNDVI; локальное изменение VH.

Отражение Sentinel-2 делится на 10 000. Sentinel-1 хранится как dB ×100 и возвращается в dB. SCL и landcover являются кодами категорий, а не отражением.

### 4.3. NaN, облака и NoData

Политика различает качество наблюдения и целевую разметку:

- `NaN/Inf` входных признаков детерминированно заменяются нулём;
- AF `valid` и BS SCL остаются признаками;
- `observation_valid` возвращается API как информационная маска и не обнуляет конкурсное предсказание;
- облачные и условно невалидные наблюдения входят в train loss и official-style validation;
- исключение из loss возможно только при реальной целевой метке 255.

GeoTIFF-маски объявляют NoData=255 в metadata, но фактически train содержит только AF `{0,1}` и BS `{0,1,2,3}`. Target=255 в проверенном train не обнаружен. Поддержка 255 в reader и loss является защитным контрактом.

## 5. Модельные решения

### 5.1. LightGBM — полностью измеренный базовый кандидат

Оба модуля используют попиксельную классификацию: learning rate 0,06; 63 листа; `min_data_in_leaf=80`; feature/bagging fraction 0,8; L2=1,0; seed 42; два CPU-потока. Внутри каждого чипа выборка ограничена по классу, затем веса восстанавливают полный train prior. AF использовал 407 955 выбранных пикселей, BS — 901 049.

AF-порог и BS-множители выбираются только на validation. Собранный и проверенный комплект V1 находится в `artifacts/tree-v1/model_bundle.zip`; manifest содержит feature version, рецепт и SHA-256 модели/metadata. V1 сохранён как резервный комплект; реальный откат сервиса к нему проверен, затем восстановлен V2.

Offline smoke на двух AF и двух BS чипах при заблокированной сети сохранён в [`artifacts/tree-v1/demo_smoke/report.json`](../artifacts/tree-v1/demo_smoke/report.json). Полный инференс V1 по фактическому test завершён за **466,8548 с**: обработано 269 чипов, сформировано 447 строк, RLE round-trip прошёл. Аудит — `artifacts/tree-v1/evidence/submission.audit.json`. Это доказывает техническую полноту V1 submission, но не качество без скрытой разметки и не скорость на инфраструктуре организаторов.

### 5.2. CNN

CNN обучается с нуля, без внешних весов. Для AF используются weighted BCE/focal + Dice, для BS — weighted cross-entropy + Dice. Normalization оценивается только по train, seed 42, validation не аугментируется.

AF CNN завершил 19 эпох, лучшая — 18-я. Единый standalone FP32 evaluation дал \(F1_{AF}=0.9045743714\) при пороге 0,826532. Полный отчёт находится в [`artifacts/cnn-v1/af/validation_report.json`](../artifacts/cnn-v1/af/validation_report.json). BS CNN завершил 20 эпох, лучшая — 13-я. После FP32-оценки и одинакового координатного tuning одиночная BS CNN дала IoU burn 0,5242176722 и mIoU severity 0,4570907406; в одиночку она уступает LightGBM.

### 5.3. XGBoost и AF-ансамбль

Предварительный AF XGBoost дал \(F1_{AF}=0.89310873\), хуже AF LightGBM/CNN, и исключён из текущего выбора. Лучший AF-рецепт смешивает **25 % CNN и 75 % LightGBM**: \(F1_{AF}=0.9295086970\), порог 0,752546. Все веса ансамбля и порог выбраны только на тех же 84 validation-чипах.

Собранный и проверенный V2 объединяет этот AF-ансамбль с tuned BS LightGBM. Рецепт — [`artifacts/ensemble-v2/selected_recipes.json`](../artifacts/ensemble-v2/selected_recipes.json). Архив [`official-ensemble-v2.zip`](https://firewatch.151.247.25.189.nip.io/downloads/official-ensemble-v2.zip) имеет размер **7 600 262 байта** и SHA-256 `7e2ec96a3bc3fe39df8d537e9593128606736438c39717ae25a9e2c4bebb00f7`.

Offline CPU smoke V2 на четырёх сценах прошёл при заблокированной сети; доказательство — [`artifacts/ensemble-v2/demo_smoke/report.json`](../artifacts/ensemble-v2/demo_smoke/report.json). Полный offline test inference V2 завершён: 269 чипов, 447 строк данных и заголовок, 540,3043 секунды по таймеру CLI при параллельном обучении. Независимый CSV validator подтвердил точный порядок шаблона, корректность RLE и отсутствие пересечений BS-классов. V2 был активирован в публичном HTTPS-сервисе, затем заменён проверенным V3; все четыре raw-маски Ubuntu/Python 3.10/PyTorch 2.6 CPU побайтно совпали с Windows-эталоном. Проверены шесть реальных запросов, площади, GeoJSON и откат. BS CNN обучена и оценена; сравнение BS XGBoost также завершено, он оставлен диагностическим экспериментом (раздел 6.2). Test не используется для выбора рецепта.

### 5.4. Выбранный V3

В окончательном сравнении BS проверены доли CNN `[0,0.25,0.5,0.75,1]`, с тем же координатным подбором множителей для новых рецептов. Лучший validation-рецепт — **25 % CNN + 75 % LightGBM**, множители `[1,2.2,2.2,1.3]`. IoU burn равен **0,5866887014**, mIoU severity — **0,5883615062**. С прежним AF это даёт Score **0,7071775413**, на **0,0044619639** выше V2. Таблица всех кандидатов: [`BS comparison`](../artifacts/cnn-v1/bs/comparison.json).

Парный bootstrap по 45 BS-чипам, 2 000 повторов и seed 42 дал percentile 95%-интервал разности weighted Score **[-0,0013900538; 0,0112153087]**. Интервал включает ноль; он условен на выбранных рецептах и не учитывает selection bias. Выбор V3 основан на наблюдаемой validation-метрике, а не на доказанном превосходстве на неизвестном private test. Подробности: [`BS CNN bootstrap`](../artifacts/cnn-v1/bs/bootstrap_comparison.json).

На Windows/Python 3.13/PyTorch 2.6 CPU V3 прошёл четыре реальные сцены при заблокированной сети. Две BS-сцены заняли 3,63–3,73 с; это небольшой локальный smoke, не конкурсный benchmark. Portable model ZIP имеет размер **14 366 427 байт**, SHA-256 `bff193672985cbdc5e74a793d91d2224cd887965c6e4497b656523997cac2880`. V2 остаётся резервом. Полный CLI, CPU replay и сервисная приёмка V3 завершены успешно.

Фактический test из **269 чипов** обработан за **390,7031 с** на T4 при параллельной CPU-проверке сохранённых весов. Получены заголовок и **447 строк данных**. Независимый [`CSV validator`](../artifacts/ensemble-v3/evidence/submission_validation.json) подтвердил точный порядок официального шаблона, канонический RLE и отсутствие пересечений BS-классов. SHA-256 `submission.csv`: `d5ba983965a146a68195bae8a38d7e53ae21fdefdf3709aade977f4eccfb739c`. GPU execution и отсутствие сетевых обращений зафиксированы отдельно; это не изолированный скоростной benchmark.

Публичный сервис использует тот же V3. Все четыре серверные и четыре скачанные публичные raw-маски побайтно совпали с локальными CPU-эталонами. Пройдены 30 сервисных тестов, 91 HTTP-проверка на шести заданиях, браузерные сценарии, сохранение отверстия полигона, реальный откат к V2 и восстановление V3. Итог: [`V3 service summary`](../artifacts/service/public-v3-release-summary.json).

## 6. Baseline и эксперименты

Официальные материалы отмечают ограничения baseline: фиксированные dNBR-пороги на разнородном покрове и отсутствие фильтрации техногенных термоаномалий. Локальный численный результат выданного baseline отсутствует, поэтому прирост относительно него не заявляется.

Наш **dNBR 4-class comparator** использует пороги 0,1/0,2/0,4 на том же validation. Это контролируемая отправная точка, но не официальный baseline.

| Метод | Стадия | \(IoU_{burn}\) | \(mIoU_{sev}\) | IoU-1 | IoU-2 | IoU-3 |
|---|---|---:|---:|---:|---:|---:|
| dNBR comparator | зафиксирован | 0,363815 | 0,354267 | 0,206089 | 0,350268 | 0,506443 |
| LightGBM, исходный recipe | собранный проверенный V1 | 0,549447 | 0,554863 | 0,351708 | 0,577234 | 0,735646 |
| LightGBM, tuned postprocess | validation-selected, включён в собранный V2 | **0,576754** | **0,585079** | **0,435736** | **0,588928** | 0,730571 |

Исходный LightGBM лучше dNBR на +0,185632 по burn IoU и +0,200596 по severity mIoU. Это поддерживает многоканальный подход, но не изолирует вклад SAR, SCL или landcover; для этого нужны отдельные абляции.

### 6.1. Постобработка BS и её цена

В [`artifacts/tree-v1/tuned_recipes.json`](../artifacts/tree-v1/tuned_recipes.json) записаны 48 последовательных координатных предложений на сохранённых validation probabilities: два прохода по классам 1/2/3, сетка значений `[0.4,0.6,0.8,1.0,1.3,1.7,2.2,3.0]`. За шаг меняется один множитель; принимается только улучшение более `1e-12`, при равенстве сохраняется текущий вариант. Это не полный перебор декартова произведения. Множители изменены с `[1.0, 1.3, 1.3, 0.7]` на `[1.0, 3.0, 3.0, 0.8]`.

AF сравнивает веса CNN `[0,0.25,0.5,0.75,1]`. Для каждого рецепта порог выбирается по полному micro-F1 среди уникальных validation scores, с правилом предсказания `score >= threshold`. Внешний validation/test для этого выбора не используется.

По confusion matrix бинарной гари:

| Recipe | Precision burn | Recall burn | IoU burn |
|---|---:|---:|---:|
| исходный | 0,6923 | 0,7270 | 0,5494 |
| tuned | 0,6163 | 0,8998 | 0,5768 |

Тюнинг сократил FN гари с 323 865 до 118 840, но увеличил FP с 383 330 до 664 494. Это осознанный сдвиг к полноте. IoU класса 1 вырос на 0,0840, класса 2 — на 0,0117, класса 3 снизился на 0,0051.

Эти показатели оптимистичны: 48 вариантов выбраны на том же validation, по которому сообщается результат. Поэтому tuned recipe — **эксперимент postprocessing с multiple-selection bias**, а не независимая оценка. Честная оценка обобщения возможна только на закрытом test или дополнительном nested holdout.

### 6.2. Дополнительный XGBoost

BS XGBoost обучен на том же train split; внешние данные и веса не использовались. Сравнены доли XGBoost 0/0,25/0,5/0,75/1 в смеси с LightGBM, с одинаковым координатным подбором множителей классов для новых кандидатов. Лучший из этих экспериментов — смесь 50/50 с множителями `[1,3,3,1]`: IoU burn 0,5785591443, mIoU severity 0,5855799928. При неизменном AF combined Score равен 0,7034977423, на 0,0007821649 выше V2. Результаты: [`comparison.json`](../artifacts/xgb-v1/bs/comparison.json).

Парный bootstrap по 45 BS validation-чипам (2 000 повторов, seed 42, суммирование confusion перед вычислением micro-метрик) дал percentile 95%-интервал разности взвешенного score **[-0,0016604721; 0,0035933433]**. Эта диагностика условна на уже выбранных рецептах, предполагает приблизительную независимость событий и **не учитывает selection bias**. Доказательство: [`bootstrap_comparison.json`](../artifacts/xgb-v1/bs/bootstrap_comparison.json).

XGBoost оставлен диагностическим экспериментом: наблюдаемая прибавка мала и не даёт убедительных оснований добавлять второй BS backend, зависимость и вычисления в проверенную поставку. Этот эксперимент не включён ни в резервный V2, ни в выбранный V3.

## 7. Результаты текущего кандидата

Источники: финальный V3 — [`artifacts/ensemble-v3/selected_recipes.json`](../artifacts/ensemble-v3/selected_recipes.json); базовый LightGBM — [`artifacts/tree-v1/validation_report.json`](../artifacts/tree-v1/validation_report.json), AF CNN/ансамбли — [`artifacts/cnn-v1/af/validation_report.json`](../artifacts/cnn-v1/af/validation_report.json), V2 recipe — [`artifacts/ensemble-v2/selected_recipes.json`](../artifacts/ensemble-v2/selected_recipes.json). Все значения — validation, не public/private test.

| Компонент | V1: LightGBM | V2: AF ensemble + tuned BS | Выбранный V3: ансамбли AF и BS |
|---|---:|---:|---:|
| \(F1_{AF}\) | 0,902135 | 0,929509 | **0,929509** |
| \(IoU_{burn}\) | 0,549447 | 0,576754 | **0,586689** |
| \(mIoU_{severity}\) | 0,554863 | 0,585079 | **0,588362** |
| Combined Score | 0,6745124420 | 0,7027155774 | **0,7071775413** |

V1 AF threshold равен 0,795563; BS multipliers — `[1.0,1.3,1.3,0.7]`. V2 AF ensemble threshold равен 0,752546; tuned BS использует `[1.0,3.0,3.0,0.8]`. V3 сохраняет AF-порог; BS использует ансамбль 25:75 и `[1,2.2,2.2,1.3]`.

| Эксперимент | Результат | Честная стадия |
|---|---:|---|
| LightGBM AF/BS V1 | 0,674512 combined | полный validation; комплект собран и test CSV проверен |
| Tuned BS postprocess | 0,693135 combined с AF tree | 48 вариантов на том же validation |
| CNN AF standalone | FP32 F1 0,904574 | полный validation |
| AF 25 % CNN + 75 % LightGBM, tuned BS | **0,702716 combined** | V2 проверен и сохранён для отката; CPU parity, полный offline test и CSV validator пройдены |
| CNN BS standalone после calibration | IoU burn 0,524218; mIoU severity 0,457091 | 20 эпох, best 13; полный FP32 validation |
| BS 25 % CNN + 75 % LightGBM, AF V2 | **0,707178 combined** | V3 развёрнут; полный test CLI, CSV validator, CPU replay и сервисная приёмка пройдены |
| XGBoost AF | F1 0,893109 | предварительно хуже; исключён |
| XGBoost BS + LightGBM, 50/50 | 0,703498 combined с AF V2 | эксперимент завершён; в поставку не включён |

Ни одно из этих чисел не является private test score. Все сравнения завершены на прежней validation-выборке. V3 заменил V2 после повторения инженерных проверок; V2 сохранён для отката.

## 8. Анализ ошибок и ограничения

1. **Класс 1 остаётся самым трудным.** В V3 IoU-1 равен 0,453508, ниже IoU-2 0,591790 и IoU-3 0,719786. Возможная причина — переходная граница между слабым повреждением, сухой растительностью и сельхозизменениями; без карт ошибок это гипотеза. По сравнению с V2 классы 1/2 улучшились, класс 3 снизился с 0,730571.
2. **Postprocessing меняет тип ошибки.** V3 даёт burn precision 0,612970 и recall 0,931897: TP 1 105 494, FP 698 010, FN 80 790. Относительно V2 пропусков меньше, ложных пикселей больше. Это осознанный компромисс по официальной метрике; визуальное предпочтение карты не использовалось для выбора модели.
3. **AF чаще даёт ложные срабатывания, чем пропуски.** Posthoc micro-подсчёт на неизменном validation даёт 144 FP против 87 FN, precision 0,913617 и recall 0,945963; это описание текущего кандидата, а не новый критерий выбора.
4. **BS validation-метрика различается по группам облачности.** В [`V3 subgroup_diagnostics.json`](../artifacts/ensemble-v3/subgroup_diagnostics.json) micro IoU burn снижается с 0,631622 при `cloud_frac <= 0.1` (33 чипа) до 0,512833 при `0.1 < cloud_frac <= 0.3` (10 чипов) и 0,401926 при `cloud_frac > 0.3`; в последней группе только 2 чипа, а все разрезы рассчитаны posthoc на том же validation и не доказывают причинную связь или переносимость.
5. **AF split изолирует географию, но не время.**
6. **Малый train** ограничивает редкие сочетания сезона, покрова и типа пожара.
7. **Score модели не калиброван как вероятность.**
8. **Наблюдавшийся runtime не является замером организаторов.** V1 обработал полный фактический test за 466,8548 с, V2 — за 540,3043 с при параллельном обучении; V3 — за 390,7031 с на T4 при параллельном CPU replay. Все сформировали валидные 447 строк данных. Условия нагрузки различались, поэтому прямой вывод об ускорении не делается. Скоростной балл определяют на инфраструктуре организаторов.

## 9. Воспроизводимость

Основные команды:

```bash
python -m competition.prepare --data-dir data/train --output prepared --split artifacts/split.json
python train.py --records prepared/records.json --output runs/cnn-v1 --task af --epochs 80 --batch-size 8 --time-limit-min 14 --workers 1 --seed 42
python train.py --records prepared/records.json --output runs/cnn-v1 --task bs --epochs 120 --batch-size 2 --time-limit-min 43 --workers 1 --seed 42
python -m competition.tree --records prepared/records.json --output runs/tree-v1 --task af --max-pixels 800000 --val-pixels 200000 --rounds 700 --num-threads 2 --seed 42
python -m competition.tree --records prepared/records.json --output runs/tree-v1 --task bs --max-pixels 1000000 --val-pixels 250000 --rounds 700 --num-threads 2 --seed 42
python -m competition.evaluate --records prepared/records.json --output runs/eval-tree --tree-root runs/tree-v1 --cache-dir runs/eval-tree/cache
python -m ops.tune_postprocess --cache runs/eval-tree/cache/bs_validation_probabilities.npz --recipe-json runs/eval-tree/selected_recipes.json --output-json runs/eval-tree/tuned_recipes.json
python -m competition.evaluate --records prepared/records.json --output runs/eval-cnn-tree --cnn-root runs/cnn-v1 --tree-root runs/tree-v1 --cache-dir runs/eval-cnn-tree/cache
python -m ops.select_bundle --recipes artifacts/ensemble-v3/selected_recipes.json --output model_bundle --af-cnn-dir runs/cnn-v1/af --af-tree-dir runs/tree-v1/af --bs-cnn-dir runs/cnn-v1/bs --bs-tree-dir runs/tree-v1/bs --bundle-id official-ensemble-v3
python inference.py --data-dir /path/to/test --model-dir /path/to/model_bundle --output submission.csv
```

Финальная команда сборки использует **зафиксированный V3 recipe**, а не автоматически максимальный результат нового запуска. SHA-256 этого recipe: `ee55b4120f7ad924af8163c81a159482d52d7bb357158f9d29dd1a93436c26b8`. В нём объединены AF-рецепт из отдельного FP32 CNN/tree evaluation и BS-ансамбль из сравнения CNN/LightGBM с координатным tuning. Полный `evaluate` также оценивает BS CNN; его coarse search по множителям не заменяет последующий `tune_postprocess`/`compare_tree_candidates`.

Мягкий лимит CNN проверяется между эпохами, после предварительного расчёта normalization. Поэтому число выполненных эпох зависит от нагрузки и оборудования. AF завершил 19 эпох и выбрал checkpoint эпохи 18; BS завершил 20 эпох и выбрал checkpoint эпохи 13. Нельзя просто заменить `--epochs 80` на 19 и ожидать те же веса: `epochs` задаёт также горизонт cosine scheduler. Seeds выставлены, но строгий deterministic-режим PyTorch не включён; побитовая повторяемость обучения не заявляется.

Наблюдавшиеся окружения зафиксированы в [`docs/environment/`](environment/):

- Colab training: Python 3.13.15, Linux, Tesla T4, exact package versions;
- Windows CPU smoke: Python 3.13.5, Windows 11, exact package versions.

Для установки служат `requirements-colab-training.txt` и `requirements-windows-cpu.txt`; `training-packages-observed.txt` — инвентаризация среды, а не инструкция установить все пакеты Colab. Замороженный исходный архив обучения `artifacts/source-training-v1.zip` имеет SHA-256 `a5cab9e409aaba69543fecb692e20a16d1f69e2ba983198fea7e090e2054eb03`. `ops.verify_training_source` подтвердил идентичность AST семи модулей обучения, признаков и метрик между этим архивом и поставляемым кодом; изменилось только форматирование. Доказательство: `artifacts/ensemble-v3/evidence/training_source_provenance.json`. Финальный архив дополнительно содержит `DELIVERY_MANIFEST.json` с SHA-256 каждого файла.

Подготовка записывает SHA-256 split. Bundle фиксирует feature version, backend, postprocessing и SHA-256 весов/metadata. CLI проверяет множество `(chip_id,class_id)`, RLE round-trip и атомарно пишет CSV.

Повторная оценка сохранённого V3 через производственный runtime на CPU выполнена по всем 129 validation-чипам (84 AF и 45 BS) при заблокированной сети. Все три компонента метрики точно совпали с исходной GPU/LightGBM-оценкой: абсолютные отклонения F1 AF, IoU гари и mIoU severity равны 0, что укладывается в допуск 0,005. Score равен 0,7071775413209714. Доказательство — [`CPU validation replay`](../artifacts/ensemble-v3/evidence/cpu_validation_replay.json). Это проверка воспроизведения предсказаний сохранённых весов, а не повторное обучение с нуля; повтор обучения отдельно не выполнен.

## 10. Источники и лицензии

Сырые данные организаторов предназначены для соревнования и не включаются в репозиторий. Внешние предобученные веса и внешняя разметка не использовались.

| Источник | Роль | Условия |
|---|---|---|
| [Copernicus Sentinel-1/2](https://dataspace.copernicus.eu/terms-and-conditions) | SAR и оптика | free, full and open; Sentinel Data Legal Notice и атрибуция |
| [NASA Earthdata / VIIRS](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) | I1–I5 | NASA-led data без отдельного ограничения — CC0; рекомендуется атрибуция |
| [ESA WorldCover v200](https://esa-worldcover.org/en/data-access) | landcover | CC BY 4.0, обязательна атрибуция |
| [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf) | DEM, slope | специальная ESA User Licence для GLO-30 |
| [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land) | t2m, rh2m, wind | CC BY; DOI 10.24381/cds.e2161bac |

Прямые программные зависимости:

| Компонент | Лицензия |
|---|---|
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt) | BSD-3-Clause |
| [Rasterio](https://github.com/rasterio/rasterio/blob/main/LICENSE.txt) | BSD-3-Clause |
| [LightGBM](https://github.com/lightgbm-org/LightGBM) | MIT |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style |
| [XGBoost](https://github.com/dmlc/xgboost/blob/master/LICENSE) | Apache-2.0 |
| [pytest](https://github.com/pytest-dev/pytest/blob/main/LICENSE) | MIT |
| [FastAPI](https://github.com/fastapi/fastapi/blob/master/LICENSE) | MIT |
| [Uvicorn](https://github.com/Kludex/uvicorn/blob/main/LICENSE.md) | BSD-3-Clause |
| [Pydantic](https://github.com/pydantic/pydantic/blob/main/LICENSE) | MIT |
| [Shapely / GEOS](https://github.com/shapely/shapely) | BSD-3-Clause / LGPL-2.1 |
| [pyproj](https://github.com/pyproj4/pyproj/blob/main/LICENSE) | MIT |
| [Leaflet](https://github.com/Leaflet/Leaflet/blob/main/LICENSE) | BSD-2-Clause |

Транзитивные лицензии поставляются в metadata установленных пакетов и должны сохраняться в финальном образе/архиве.

## 11. Проверки поставки и оставшиеся ограничения

- обучение и сравнение LightGBM/CNN/XGBoost завершены; выбран V3, V2 сохранён как проверенный резерв;
- локальный offline CPU smoke V3 выполнен на четырёх реальных сценах;
- полный test CLI 269 чипов и независимый CSV validator 447 строк пройдены; CPU replay всех 129 validation-чипов дал нулевые отклонения по трём метрикам;
- серверная приёмка V3 завершена: восемь проверок, включая публичные маски, браузер и реальный rollback/restore, пройдены;
- повторное обучение с нуля в допуске 0,005 отдельно не выполнено; проверен replay сохранённых весов;
- закрытый test Score и скоростные баллы определяют организаторы;
- будущие абляции нужны для отдельного измерения вклада SAR, SCL и landcover; текущие эксперименты не изолируют их причинный вклад.

Числа раздела 7 являются validation-результатами. Они не подтверждают независимое качество или победу в соревновании.
