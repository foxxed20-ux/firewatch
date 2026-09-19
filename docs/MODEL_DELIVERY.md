# Использование конкурсного архива

Архив содержит автономные модели AF/BS, код обучения и инференса, выбранные веса в `model_bundle`, готовый `submission.csv`, конфигурации, технический отчёт и материалы для жюри. `DELIVERY_MANIFEST.json` фиксирует SHA-256 каждого включённого файла.

Для запуска моделей используйте `README_MODEL.md`:

```bash
pip install -r requirements-model.txt
python inference.py --data-dir /path/to/test --output submission.csv
```

Зависимости устанавливаются заранее; сам инференс работает без сети. Сырые конкурсные снимки в архив не включены. Получите их по ссылке организаторов и сверяйте с `artifacts/data_source_lock.json`. Зафиксированный train/validation split находится в `artifacts/split.json`.

Технический отчёт: `docs/MODEL_REPORT.md`. Паспорт проекта и исследование: `output/pdf/`. Презентация: `docs/presentation/FireWatch_КосмоХакатон_2026.pptx`.

Корневой `README.md`, `README_SERVICE.md` и `MODEL_API.md` описывают весь проект. Для самостоятельного развёртывания веб-сервиса нужен полный исходный репозиторий: [GitVerse](https://gitverse.ru/hackrus.experts/kosmo-krasnoiarsk_4_th_try_106), [зеркало GitHub](https://github.com/foxxed20-ux/firewatch). Публичная демонстрация: [FireWatch](https://firewatch.151.247.25.189.nip.io/).

Метрики отчёта получены на validation, использованном также для выбора модели и постобработки. Закрытый test Score неизвестен. Проверка сохранённых весов на CPU выполнена отдельно; повтор обучения с нуля не заявляется проверенным.
