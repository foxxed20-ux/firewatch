# Google Maps и Earth Engine: настройка FireWatch

Инструкция описывает ручную подготовку Google Cloud для спутниковой подложки Google Maps в браузере и поиска/предпросмотра реальных Sentinel-2 сцен через Google Earth Engine (EE). Она не создаёт проекты, ключи, identities или платные ресурсы. Эти действия выполняет владелец Google Cloud проекта.

Без Google приложение остаётся рабочим: конфигурация сообщает состояние настройки, а EE-маршруты возвращают явную ошибку, а не вымышленные сцены.

## Граница функции

EE используется только для metadata и временного RGB tile URL сцен из `COPERNICUS/S2_SR_HARMONIZED`; свойства коллекции приведены в [официальном каталоге](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED). Нет экспорта растров, создания EE assets или загрузки результатов в каталог модели.

- AF по-прежнему требует VIIRS и вспомогательные данные.
- BS по-прежнему требует Sentinel-2 до/после, Sentinel-1 и вспомогательные данные.
- Поиск/preview помогает выбрать и проверить данные, но не обещает прогноз.

| Компонент | Получает | Никогда не получает |
|---|---|---|
| Браузер (`web/`) | Maps browser API key | EE OAuth/ADC, JSON-ключ service account |
| API FireWatch | Cloud project ID и серверные ADC/credential path | browser key для EE |
| Earth Engine | аутентифицированные серверные вызовы | OAuth credential в браузере |

Maps key виден посетителю страницы: его ограничивают referer и API. Серверный credential — секрет: не помещайте его в HTML, JavaScript, Git, CI-логи, ticket или API-ответ.

## Решения владельца Cloud

1. Выберите существующий Google Cloud project или отдельный project FireWatch. Запишите его **project ID**, не display name.
2. Выберите commercial/noncommercial регистрацию EE по реальной применимости. Открытые данные или публичный сайт сами по себе не определяют статус. Прочтите [EE access and registration](https://developers.google.com/earth-engine/guides/access).
3. Production Maps JavaScript API требует billing и standard API key; Maps Demo Key предназначен только для прототипов. См. [Maps usage and billing](https://developers.google.com/maps/documentation/javascript/usage-and-billing).

EE registration и Maps setup могут требовать платёжный профиль или вызывать расходы. Эта инструкция не утверждает применимость некоммерческого режима и не выполняет платных действий.

## Ручная настройка в Cloud Console

### Earth Engine

1. Войдите владельцем/администратором project и откройте [EE registration page](https://code.earthengine.google.com/register). Это рекомендуемый Google способ создать либо зарегистрировать Cloud project.
2. Проверьте, что включён **Earth Engine API**. Официальные ручные шаги есть в [EE access guide](https://developers.google.com/earth-engine/guides/access#enable-the-earth-engine-api).
3. Завершите registration как commercial или noncommercial по фактическому случаю. Cloud project обязателен для EE с 13 ноября 2024: [migration guidance](https://developers.google.com/earth-engine/guides/transition_to_cloud_projects).
4. Для серверной identity, читающей публичную collection и создающей EE computations, выдайте минимальные роли **Earth Engine Resource Viewer** (`roles/earthengine.viewer`) и, если это требуется конфигурацией, **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`). Эти роли приводит Google для service account с EE REST computations: [service account guide](https://developers.google.com/earth-engine/guides/service_account#setup-rest-api-access). Канонический ID роли Viewer указан в [справочнике Google IAM](https://docs.cloud.google.com/iam/docs/roles-permissions/earthengine#earthengine.viewer); в Console название может содержать пометку Beta. Не выдавайте Owner, Editor, Resource Writer или Service Account Admin только ради поиска публичных сцен.

Registration, enabled API и IAM — отдельные prerequisites. Одного project ID недостаточно.

Для RGB preview нужно дополнительное разрешение **`earthengine.maps.create`**:
его требует [метод создания карты](https://developers.google.com/earth-engine/reference/rest/v1/projects.maps/create).
В `roles/earthengine.viewer` этого разрешения нет: поиск может работать, а preview
возвращать `EARTH_ENGINE_UNAVAILABLE` из-за upstream permission denied. Владелец
проекта может добавить это разрешение отдельной custom role, проверив его
доступность в редакторе IAM; [порядок создания custom role](https://docs.cloud.google.com/iam/docs/creating-custom-roles).
На 19.09.2026 поддержка `earthengine.maps.create` в custom roles имеет уровень
**TESTING**: Google допускает добавление, но предупреждает о возможном
неожиданном поведении и не рекомендует такое разрешение для production.
Это подтверждает [справочник уровней поддержки](https://docs.cloud.google.com/iam/docs/custom-roles-permissions-support?hl=en).
Согласование для прототипа не означает гарантии стабильности или SLA; перед
production-развёртыванием отдельно проверяют поддержку выбранной IAM-конфигурации.
Готовая роль `roles/earthengine.writer` также содержит `earthengine.maps.create`,
но даёт более широкие права, включая запись и удаление assets; выбор такой роли
нужно отдельно согласовать с владельцем проекта. После настройки проверьте и
`search`, и `preview`, и загрузку реального tile.

### Google Maps satellite basemap

1. В выбранном project включите **Maps JavaScript API**.
2. Создайте отдельный key, например `firewatch-web-maps`; не используйте его для серверных вызовов.
3. В *Application restrictions* выберите **Websites (HTTP referrers)** и внесите только реальные origins:

   ```text
   https://firewatch.example/*
   https://firewatch.151.247.25.189.nip.io/*
   http://localhost:8089/*       # только локальная разработка
   ```

   Удалите localhost перед публичной поставкой, если он уже не нужен. Адрес обязан совпадать со схемой и origin страницы. `RefererNotAllowedMapError` означает, что текущий URL не разрешён: [ошибки Maps](https://developers.google.com/maps/documentation/javascript/error-messages).
4. В *API restrictions* выберите **Restrict key** и оставьте только **Maps JavaScript API**. Google рекомендует ограничивать ключ и по приложению, и по API: [setup Maps key](https://developers.google.com/maps/documentation/javascript/get-api-key).
5. Не подменяйте licensed Maps tiles сторонними запросами; сохраняйте штатную attribution. См. [Maps JavaScript API policies](https://developers.google.com/maps/documentation/javascript/policies).

## Серверная identity Earth Engine

Предпочтителен **Application Default Credentials (ADC)**: на Google runtime — attached service account, в иной среде — согласованный workload identity. Google рекомендует ADC для unattended среды, так как не нужен постоянный JSON private key: [EE service accounts](https://developers.google.com/earth-engine/guides/service_account#authenticate-to-earth-engine-using-application-default-credentials).

Для локальной разработки допустим пользовательский ADC:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Это developer credential, не production identity. Общий порядок ADC описывает [Google Cloud authentication](https://cloud.google.com/docs/authentication/provide-credentials-adc).

Если attached/workload identity невозможен, допустим отдельный service account с ролями выше и JSON-key вне Git/релиза, доступный лишь владельцу процесса:

```text
GOOGLE_APPLICATION_CREDENTIALS=/secure/firewatch/ee-service-account.json
```

Это серверный путь, не значение для `.env.example`. Не публикуйте key file; Google требует хранить его безопасно и отозвать при утрате: [key protection](https://developers.google.com/earth-engine/guides/service_account#authenticate-with-a-private-key).

## Конфигурация FireWatch

Production environment хранится вне Git и загружается service manager. Шаблон никогда не содержит реальный ключ или путь к credential.

Сервер использует закреплённый `earthengine-api==1.7.43` (Python 3.10+). При
обновлении SDK сначала проверяют контракт injected-provider и отдельную живую
интеграцию после появления credentials; этот документ не считает live-проверку
выполненной.

При холодном старте backend передаёт `httplib2.Http(timeout=...)` в
`ee.Initialize`, отключает повторы SDK до инициализации и вызывает
`ee.data.setDeadline` после неё. В SDK 1.7.43 вызов `setDeadline` до
`Initialize` завершается ошибкой; транспорт ограничивает время сетевого
ожидания уже при загрузке API и обновлении credentials.

```dotenv
# Видим браузеру; ограничен HTTP referrers и Maps JavaScript API.
FIREWATCH_GOOGLE_MAPS_API_KEY=
# Project ID с EE registration и включённым Earth Engine API.
FIREWATCH_GOOGLE_CLOUD_PROJECT=
# Без явного true EE-маршруты выключены.
FIREWATCH_EARTH_ENGINE_ENABLED=false
# Необязательный protected path вне Google runtime:
# GOOGLE_APPLICATION_CREDENTIALS=/secure/firewatch/ee-service-account.json
FIREWATCH_IMAGERY_MAX_AREA_KM2=10000
FIREWATCH_IMAGERY_MAX_DATE_DAYS=366
FIREWATCH_IMAGERY_TIMEOUT_SECONDS=20
FIREWATCH_IMAGERY_RATE_LIMIT_PER_MINUTE=10
FIREWATCH_IMAGERY_CACHE_SECONDS=300
FIREWATCH_IMAGERY_CACHE_ENTRIES=128
FIREWATCH_IMAGERY_MAX_CONCURRENT=2
```

`FIREWATCH_IMAGERY_RATE_LIMIT_PER_MINUTE` — общий limiter одного Uvicorn process, дополняющий существующий per-IP limiter. При нескольких processes понадобится общий внешний счётчик. Area/date/concurrency/timeout/cache — предохранители API от дорогих и зависающих интерактивных запросов, не Google quota. Cache TTL не превышает 300 секунд; cache hit не создаёт нового вызова EE и не расходует дополнительный Google request budget.

Gateway допускает до двух незавершённых обращений по умолчанию. У SDK Earth
Engine общее состояние и HTTP-клиент, поэтому production provider выполняет
только один SDK-вызов одновременно: второй сразу получает `IMAGERY_BUSY`,
без внутренней очереди. После HTTP-таймаута уже начатый вызов может ещё
завершаться; он удерживает слот до фактического окончания. Это предотвращает
накопление новых вызовов при повторных запросах клиента.

## Контракт и состояния API

`GET /v1/client-config` публичен и возвращает только безопасные значения:

```json
{
  "maps": {"provider": "google", "api_key": "browser-key", "configured": true},
  "imagery": {"provider": "earth_engine", "configured": true,
              "collection": "COPERNICUS/S2_SR_HARMONIZED"}
}
```

`configured` означает локальные prerequisites/flag, но не live credential, quota, billing или наличие сцен. Без Maps key API возвращает `provider: "osm"` и `api_key: null`; без EE — `configured: false` и безопасную причину.

Основная авторизация сервиса действует и для:

```text
POST /v1/imagery/search
{ "aoi": {GeoJSON Polygon|MultiPolygon}, "date_start": "YYYY-MM-DD",
  "date_end": "YYYY-MM-DD", "cloud_max": 0..100, "limit": 1..20 }
POST /v1/imagery/preview
{ "scene_id": "только ID из search", "aoi": {GeoJSON Polygon|MultiPolygon} }
```

В public demo авторизация заменяется явно включённым режимом, но global limiter остаётся. Даты включительны, UTC и их число не превышает 366. `search` возвращает реальные EE metadata: `id`, `acquired_at`, `cloud_percent`, `bounds: [west,south,east,north]`, `sensor: "Sentinel-2"`, `resolution_m: 10`, `truncated`. Для `preview` клиент берёт `scene_id` из search. Сервер проверяет разрешённую коллекцию и формат ID, повторно ограничивает AOI и возвращает `tile_url`, `attribution`, `bounds`; `expires_at` намеренно не выдаётся. Привязки к истории поисков пользователя нет. MGRS tile в Sentinel-2 имеет формат наподобие `T56MNN`. URL слоя может быть временным; OAuth credential и server access token не выдаются.

AOI допускает только WGS84 `Polygon`/`MultiPolygon`: не более 1 000 двумерных
координат, latitude в `[-85, 85]`, longitude span не более 180°. Антимеридиан
клиент предварительно делит на части. Лимит `10000 km²` вычисляется для
bounding rectangle AOI в equal-area EPSG:6933, включая holes и разнесённые
parts. Дополнительно ширина и высота этого rectangle ограничены 300 км каждая:
маленькая площадь длинной узкой полосы не позволяет искать сцены через полмира.

Используйте `GOOGLE_NOT_CONFIGURED` для выключенного/неполного EE или отсутствующего ADC,
`EARTH_ENGINE_UNAVAILABLE` для ошибки инициализации или обращения к EE, `EARTH_ENGINE_TIMEOUT`
с HTTP 504 при истечении 20 секунд, `422` для AOI/дат. Глобальный limiter
отвечает HTTP 429 `IMAGERY_RATE_LIMITED` с `Retry-After: 60`; отсутствие
свободного EE worker — HTTP 429 `IMAGERY_BUSY` с `Retry-After: 2`. Пустой
`scenes` — честный успешный поиск без сцен.

## Ручная приемка после настройки

`configured: true` не является live acceptance test.

1. После защищённого environment file перезапустите только API.
2. Проверьте `/v1/client-config`: Maps `google`, EE `configured: true`; в JSON нет EE/OAuth credential.
3. На разрешённом origin откройте satellite map: нет ошибок console, есть Google attribution. На неразрешённом origin key должен быть отклонён.
4. Выполните один авторизованный `search` для небольшой Polygon/короткого периода, затем `preview` с ID из ответа; сверьте acquisition time, cloud metadata, HTTP status и tile layer.
5. Проверьте отрицательные случаи: выключенный flag, неверный project, AOI выше bounding-rectangle лимита, >1 000 позиций, latitude/span вне пределов, период >366 дней, `limit:21`, чужой scene ID, global limiter и busy worker. Для последних двух проверьте соответственно `Retry-After: 60` и `Retry-After: 2`; timeout должен давать HTTP 504. API возвращает безопасный код ошибки; не записывайте ключи или credential JSON в журналы.
6. В Cloud Console зафиксируйте enabled APIs, browser key restrictions, IAM, EE registration, quotas и billing как ручное acceptance evidence.

Автотесты с injected fake-provider проверяют контракт, лимиты и безопасные ошибки. Они не подтверждают живой проект, EE eligibility, оплату, server identity или реальную сцену; это ручные gates до настоящей конфигурации.

## Квоты и расходы

До публичного доступа задайте budget alerts и просматривайте Maps/Cloud quota dashboard. [Cloud Billing budgets](https://cloud.google.com/billing/docs/how-to/budgets) предупреждает: alerts-only budget уведомляет, но не останавливает использование или списания. Лимиты FireWatch выше — дополнительная защита, не гарантия нулевого счёта. Автоматические действия по бюджету отдельно проектирует и утверждает владелец billing.

| Наблюдение | Проверить |
|---|---|
| `RefererNotAllowedMapError` | referrer restrictions, схему и origin страницы |
| `ApiNotActivatedMapError` | включён ли Maps JavaScript API в project этого key |
| `GOOGLE_NOT_CONFIGURED` | flag, project ID, EE registration/API |
| `EARTH_ENGINE_UNAVAILABLE` | ADC/service-account roles, credential и project |
| Нет сцен | AOI, включительные UTC даты, `cloud_max`; это не модельная ошибка |
| `IMAGERY_RATE_LIMITED` | ждите `Retry-After: 60`; не обходите global limiter |
| `IMAGERY_BUSY` | ждите `Retry-After: 2`; SDK занят или исчерпаны слоты gateway |
| `EARTH_ENGINE_TIMEOUT` | HTTP 504; уменьшите AOI/период/limit, не ослабляйте предел без измерений |

Код EE передаёт project явно: `ee.Initialize(credentials, project='YOUR_PROJECT_ID')`. [EE authentication and initialization guide](https://developers.google.com/earth-engine/guides/auth) подтверждает, что вызов проверяет credential и маршрутизирует операции через этот Cloud project.
