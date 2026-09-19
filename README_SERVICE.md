# FireWatch: запуск сервиса и работа с API

Сервис принимает область интереса и период, выполняет AF/BS-инференс на зарегистрированных сценах и возвращает карту, GeoJSON, маски и JSON-справку. AF и BS запускаются независимо. Общая библиотека `competition.service_bridge` используется также конкурсным CLI.

Публичный адрес: [FireWatch](https://firewatch.151.247.25.189.nip.io/). [Swagger UI](https://firewatch.151.247.25.189.nip.io/docs) и [OpenAPI](https://firewatch.151.247.25.189.nip.io/openapi.json) описывают текущие HTTP-запросы. Состояние модели проверяется через `/health/ready`; доступность страницы сама по себе не подтверждает готовность анализа. Фактические результаты приёмки публикуются отдельно от этой инструкции.

## Данные и границы демонстрации

Подготовлен набор `official-validation-demo`: два AF-чипа и два BS-чипа из открытого train организаторов, отнесённых фиксированным разбиением к validation. Примеры выбраны для показа, поэтому результаты на них не являются независимой оценкой качества. Исходные координаты, даты и происхождение сохранены. Обезличенный test в геокаталог не импортируется.

Это конечный каталог сцен, а не загрузка произвольных спутниковых снимков по запросу. Для территории или даты без подходящих наблюдений возвращается `no_data`. Модель не заменяется искусственными предсказаниями при отсутствии весов.

Каталог и исходные TIFF не входят в публичный Git-репозиторий. Локальный импорт разрешённого input-only архива:

```powershell
python tools/prepare_service_demo.py --data-root data/service --import-demo-zip PATH_TO_AUTHORIZED_INPUT_ARCHIVE.zip
```

Importer сохраняет manifest в `data/service/datasets/official-validation-demo/manifest.json`. Каждый пример содержит `bounds`, `period`, `tasks` и исходные scene IDs. Manifest и растр после публикации не изменяют на месте: для новых данных создают новый dataset ID и каталог. В заданиях сохраняются ревизия manifest и выбранные сцены; контроль содержимого всех исходных TIFF при каждом запросе не выполняется.

## Локальный запуск

Фактически используемые среды: Windows/Python 3.13.5 для сервисных тестов и Ubuntu 22.04/Python 3.10.12 для сервера. Зависимости закреплены в `requirements-service.txt`.

```powershell
python -m venv .venv-service
.\.venv-service\Scripts\python.exe -m pip install -r requirements-service.txt
$env:FIREWATCH_DATA_ROOT = (Resolve-Path data/service).Path
$env:FIREWATCH_MODEL_ROOT = (Resolve-Path model_bundle).Path
.\.venv-service\Scripts\python.exe -m uvicorn firewatch_service.api:app --host 127.0.0.1 --port 8089 --workers 1
```

Команды выполняют из корня проекта: статические файлы находятся в `web/`. В Linux используют `.venv-service/bin/python`. Для bundle с CNN дополнительно нужен PyTorch; для CPU-сервера закреплён вариант в `deploy/requirements-cpu.txt`. Tree-only bundle использует LightGBM и не требует PyTorch для предсказаний.

Каталог модели должен содержать `manifest.json` и все указанные в нём веса/metadata с SHA-256. Идентификатор берут из реестра `/v1/models`, не предполагают заранее `official-v1`. Препроцессинг, порядок признаков, пороги и class multipliers задаются самим bundle.

## Настройки и доступ

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FIREWATCH_DATA_ROOT` | `data/service` | Подготовленные наборы |
| `FIREWATCH_MODEL_ROOT` | `model_bundle` | Один полный AF+BS bundle |
| `FIREWATCH_DOWNLOAD_ROOT` | отключено | Только явно опубликованные публичные архивы весов |
| `FIREWATCH_ARTIFACT_ROOT` | `var/results` | Результаты по job ID |
| `FIREWATCH_JOB_DB` | `var/jobs.sqlite3` | Статусы и идемпотентность |
| `FIREWATCH_TOKEN` | отсутствует | Bearer token закрытой установки |
| `FIREWATCH_PUBLIC_DEMO` | `false` | Явное включение публичного демонстрационного доступа |
| `FIREWATCH_MAX_WORKERS` | `1` | Исполнители; один процесс API |
| `FIREWATCH_MAX_QUEUED_JOBS` | `16` | Максимум принятых queued + running заданий |
| `FIREWATCH_JOB_TIMEOUT_SECONDS` | `900` | Лимит одного worker; на сервере 300 секунд |
| `FIREWATCH_QUEUE_TIMEOUT_SECONDS` | `300` | Максимальное ожидание в очереди |
| `FIREWATCH_MAX_POST_BODY_BYTES` | `32768` | Предел HTTP-запроса |
| `FIREWATCH_RATE_LIMIT_PER_MINUTE` | `30` | Общий лимит POST, `/v1/models`, `/health/ready` на IP |
| `FIREWATCH_REGISTRY_CACHE_SECONDS` | `10` | Кэш проверки целостности bundle |
| `DEVICE` | `auto` в model runtime | На сервере явно `cpu` |

Без токена и без публичного режима API доступен только с loopback. Для закрытой установки задают `FIREWATCH_TOKEN` и оставляют `FIREWATCH_PUBLIC_DEMO=false`; запросы используют `Authorization: Bearer ...`. Публичный режим явно отключает требование токена и предназначен для предоставленного каталога демонстрации. За reverse proxy сохраняют реальный IP, доверяя forwarded headers только своему proxy.

`GET /health/live` подтверждает работающий процесс. `GET /health/ready` требует непустой каталог, целостный совместимый bundle для задач каталога и доступный executor; первый реальный инференс отдельно проверяют перед релизом. Health endpoint не измеряет качество модели.

## Запрос анализа

Сначала получить `GET /v1/catalog` и `GET /v1/models`. Отправить `POST /v1/analyses` с `Content-Type: application/json` и обязательным `Idempotency-Key`:

```json
{
  "dataset_id": "official-validation-demo",
  "model_bundle_id": "official-tree-v1",
  "aoi": {"bbox": [40.89428609771226, 46.58422763469897, 41.03131268381209, 46.67862114951406]},
  "period": {"start": "2024-07-13T00:00:00Z", "end": "2024-07-29T00:00:00Z"},
  "tasks": ["bs"]
}
```

Версия модели в примере должна присутствовать в текущем реестре. Вместо bbox допускается `aoi.geometry` с валидным GeoJSON `Polygon` или `MultiPolygon` в WGS84; порядок координат — долгота, широта. Кольца замкнуты; самопересечения, переход через антимеридиан в bbox и слишком сложная геометрия отвергаются.

Интервал времени полуоткрытый: **`start <= t < end`**. Для AF используется время наблюдения, для BS — дата post; pre должна быть раньше post. Для включения всего дня 28 июля передают end = 29 июля 00:00 UTC. Предварительный снимок BS может предшествовать start: он нужен для сравнения, а попадание в период определяется post.

HTTP 202 возвращает `job_id`, `status_url`, `result_url`. Сохраняйте ключ идемпотентности до первого POST: повтор того же тела с тем же ключом возвращает то же задание; другое тело с этим ключом — HTTP 409. Отдельно зафиксированный worker payload содержит снимок каталога и абсолютный путь к неизменяемой модели релиза.

Полный пример PowerShell, выбирающий существующие значения из API:

```powershell
$base = 'https://firewatch.151.247.25.189.nip.io'
$headers = @{}
if ($env:FIREWATCH_TOKEN) { $headers.Authorization = 'Bearer ' + $env:FIREWATCH_TOKEN }
$catalog = Invoke-RestMethod "$base/v1/catalog" -Headers $headers
$models = Invoke-RestMethod "$base/v1/models" -Headers $headers
$dataset = $catalog.datasets | Where-Object { $_.dataset_id -eq 'official-validation-demo' } | Select-Object -First 1
$preset = $dataset.presets | Where-Object { $_.tasks -contains 'bs' } | Select-Object -First 1
$model = $models.models | Where-Object { $_.status -eq 'ready' } | Select-Object -First 1
if (-not $preset -or -not $model) { throw 'Данные или модель пока не готовы' }
$payload = @{ dataset_id=$dataset.dataset_id; model_bundle_id=$model.model_bundle_id; aoi=@{bbox=$preset.bounds}; period=$preset.period; tasks=$preset.tasks } | ConvertTo-Json -Depth 8
$headers['Idempotency-Key'] = [guid]::NewGuid().ToString()
$job = Invoke-RestMethod "$base/v1/analyses" -Method Post -Headers $headers -ContentType 'application/json' -Body $payload
$deadline = [DateTime]::UtcNow.AddMinutes(10)
do {
    if ([DateTime]::UtcNow -gt $deadline) { throw "Истекло ожидание; сохраните job_id=$($job.job_id)" }
    Start-Sleep -Seconds 2
    $state = Invoke-RestMethod ($base + $job.status_url) -Headers $headers
} while ($state.status -in @('queued','running'))
if ($state.status -ne 'succeeded') { throw ($state.error | ConvertTo-Json) }
$result = Invoke-RestMethod ($base + $job.result_url) -Headers $headers
$result.modules | ConvertTo-Json -Depth 8
$contours = $result.artifacts | Where-Object { $_.id -eq 'burn_polygons' }
Invoke-WebRequest ($base + $contours.href) -Headers $headers -OutFile contours.geojson
```

Истечение ожидания клиента не отменяет задание. После перезапуска API прерванные running задания становятся `failed/WORKER_LOST`; queued задания возобновляются. Новый повтор вычисления после terminal failure использует новый ключ. Рабочие процессы ограничены временем и завершаются при таймауте.

## Чтение результатов

- `data_status`: `complete`, `partial`, `no_data`. Успешное завершение job и полнота наблюдений — разные признаки.
- `modules.af.hotspot_count`: число термоточек в AOI. Разные пролёты сохраняются отдельно. Чипы одного пролёта могут дедуплицироваться при явных `source_scene_id`, `grid_row_offset`, `grid_col_offset`; без этих метаданных это отдельные наблюдения.
- `modules.bs.burn_area_ha` и `severity_area_ha`: общая площадь и классы `1`, `2`, `3`. При перекрытии BS более поздний валидный post скрывает более раннее наблюдение, в том числе его гарь.
- `quality.<task>`: наблюдаемая и ненаблюдаемая площадь, доля покрытия, число наблюдений. `cloud_fraction=null` означает, что самостоятельная доля облачности не рассчитана; это не ноль.
- При `no_data` показатели пожара равны `null`, покрытие — нулю. Ненаблюдаемые пиксели не считаются подтверждённым фоном. Незапрошенные модули отсутствуют.

GeoJSON записан в WGS84 и ограничен AOI. Площадь вычисляется в равноплощадной EPSG:6933 в гектарах; классы BS не пересекаются. Суммы считаются до округления контуров. `contour_id`, `class_id`, `severity`, `area_ha`, даты и bundle ID находятся в свойствах объектов. GeoJSON выполняет требование экспорта кейса; Shapefile в этой версии не формируется.

`artifacts` содержит относительные URL `hotspots`, `burn_polygons`, `summary`, `provenance`, display masks и вспомогательный `pixel_rles`. `summary` содержит те же показатели и качество, что API. GeoTIFF для карты использует NoData=255; конкурсная сырая маска сохраняется через endpoint predictions. Каждый скачиваемый файл имеет SHA-256 в HTTP ETag. Исходные входные TIFF через API не публикуются.

## Сырые маски модели

`POST /v1/predictions` с теми же заголовками:

```json
{"dataset_id":"official-validation-demo","model_bundle_id":"official-tree-v1","chip_ids":["AF_tr_000126","BS_tr_000159"]}
```

Допускаются только зарегистрированные chip IDs. В результате `items[]` содержит task, размеры, classes, `mask_url` и `rle`; общий `provenance_url` указывает происхождение. AF — uint8 0/1, BS — uint8 0/1/2/3. RLE: C-order, начало с 1, для BS отдельные непересекающиеся классы 1–3. При наличии геопривязки маска GeoTIFF, иначе NumPy. Эти сырые маски соответствуют общей модельной библиотеке; display-обработка AOI/NoData их не меняет.

## Ошибки

| HTTP | Смысл / действие |
|---|---|
| 400 | Нет корректного `Idempotency-Key` |
| 401 | Требуется авторизация закрытой установки |
| 404 | Неизвестный chip, job или артефакт |
| 409 | Конфликт идемпотентности, ещё нет результата или job failed |
| 413 | Превышен размер тела запроса |
| 422 | Некорректные AOI, даты, схема или dataset ID |
| 429 | Лимит запросов/очереди; учитывать `Retry-After`, сохранить ключ |
| 503 | Каталог/модель пока не готовы |

Ошибки вычисления находятся в `GET /v1/jobs/{job_id}`. Клиент получает безопасное сообщение, диагностическое исключение — в серверном журнале. Неизвестный результат сетевого POST повторяют с исходным ключом; не создают новый job автоматически.

## Развёртывание и откат

Приложение работает на приватном сервере в `/home/red/firewatch`, Uvicorn слушает только `127.0.0.1:8089`. Отдельный systemd SSH tunnel передаёт его на loopback публичного relay `127.0.0.1:18089`. Nginx relay обслуживает HTTPS. Шаблоны — `deploy/firewatch.service`, `firewatch-tunnel.service`, `nginx-https.conf`, `service.env.example`. Закрытый SSH-ключ и env с секретами размещают вне релиза; пароли в репозитории не хранятся.

Подготовка неизменяемого артефакта после тестов:

```powershell
python deploy/package_release.py --name RELEASE_NAME --model-root model_bundle
```

Архив в `artifacts/service/releases/` содержит код, web и полный bundle с контрольными суммами. `release.env` связывает модель с абсолютным каталогом данной версии. Для начальной диагностики без модели существует отдельный флаг `--bootstrap-without-model`; такой выпуск не считается готовым к анализу.

После переноса архива на app-сервер:

```bash
/home/red/firewatch/venv/bin/python /home/red/firewatch/deploy/verify_release.py --archive /path/to/RELEASE_NAME.tar.gz --sha256 RECEIPT_SHA256 --directory /home/red/firewatch/releases/RELEASE_NAME
/home/red/firewatch/deploy/activate_release.sh RELEASE_NAME
```

Activation проверяет файлы, меняет `current` атомарно, перезапускает unit и ждёт `/health/ready`. `check_readiness.py` читает режим доступа и токен из env-файла; токен не передаётся в аргументах процесса. При неудаче возвращается предыдущий symlink. Для явного отката выполняют ту же команду с именем ранее проверенного релиза. Код и веса возвращаются вместе; каталоги прежних релизов сохраняют, пока существуют связанные queued jobs. DB и результаты находятся в `shared/`; схема DB расширяется добавлением private worker payload, старые результаты сохраняются.

Само переключение symlink не доказывает исправность продукта: после активации выполняют реальный запрос и проверяют скачанные результаты. TLS renewal использует штатный certbot timer и nginx reload hook. Контроль места на relay и app-сервере остаётся обязанностью оператора; автоматическое удаление истории и артефактов в этой версии не включено.

## Проверка

```powershell
python -m pip install pytest httpx requests
python -m pytest tests/service -q
python tools/check_service_e2e.py --base-url https://firewatch.151.247.25.189.nip.io --output artifacts/service/UNIQUE_RUN
```

Сквозная проверка использует настоящие presets и bundle: четыре анализа, идемпотентный повтор/конфликт, сырые маски и обратное декодирование RLE, воспроизводимость площадей по GeoJSON, отсутствие пересечений классов, `no_data`, ошибки входа и SHA скачанных файлов. В каталоге запуска сохраняются фактический отчёт и продукты. Проверка интерфейса в браузере и отката выполняется дополнительно. Тесты с fixture-масками проверяют геоалгоритм и HTTP-контракты, но не являются измерением качества обученной модели.
