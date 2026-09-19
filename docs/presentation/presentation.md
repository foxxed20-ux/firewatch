# FireWatch — презентация для жюри

**[Скачать редактируемую презентацию](FireWatch_КосмоХакатон_2026.pptx)** — 10 слайдов, заметки докладчика и редактируемая диаграмма. PPTX можно открыть и изменить в PowerPoint или совместимом редакторе.

`build_presentation.mjs` сохраняет исходник сборки, а `metrics.json` — использованные показатели. Повторная сборка требует среды с `@oai/artifact-tool` и утилитами навыка Presentations; они не входят в зависимости моделей или сервиса. Укажите путь к установленному навыку в `FIREWATCH_PRESENTATIONS_SKILL_DIR`, затем из корня проекта выполните `node docs/presentation/build_presentation.mjs`. Абсолютные пути автора не требуются. Готовый PPTX не требует этой среды.

## Содержание и заметки докладчика

1. **Титульный слайд.** FireWatch связывает наблюдение активного горения и картирование последствий.
2. **Проблема.** Термоточка не заменяет контур гари. AF и BS отвечают на разные вопросы.
3. **Данные.** AF использует VIIRS, BS использует Sentinel-2, Sentinel-1 и AUX. На выходе маски и классы тяжести.
4. **Пользовательский результат.** AOI и период проходят каталог, инференс, геообработку и возвращаются на карту с экспортом.
5. **Архитектура.** CLI и сервис вызывают единый `competition.service_bridge`; сервис добавляет очередь, артефакты и API.
6. **Валидация.** Открытый train разделён на train/validation; BS дополнительно изолирует события. Признаки не используют координаты, даты или ID.
7. **Метрики.** Это результаты validation v3: AF ensemble 25% CNN и 75% LightGBM; BS ensemble 25% CNN и 75% LightGBM с multipliers `[1, 2.2, 2.2, 1.3]`. Score выше V2 на 0,00446, но bootstrap 95% `[-0,00139; +0,01122]` включает ноль, а результат условен на selected recipes. Full V3 CLI завершён на 269 чипах и 447 строках за 390,703 с на GPU T4/CUDA; CPU replay 129 validation-чипов дал Δ=0 во всех трёх checks. Независимый CSV validator завершился с EXIT 0: 447 data rows, exact order, canonical RLE и BS exclusivity. Нельзя называть это оценкой закрытого test.
8. **Демонстрация.** В интерфейсе пользователь выбирает область и период, запускает задание и получает карту. Публичный HTTPS capture V3 показывает release `20260919-ensemble-4`, bundle `official-ensemble-v3`, `BS_tr_000159`, 1 037,3 га, покрытие 25,4%, 1 171 полигон; severity 250,6 / 178,8 / 607,9 га. Это train/validation-демо, не закрытый test; browser BS и export прошли.
9. **Проверка.** V3 сформировал полный CLI CSV на 269 чипах и 447 строках за 390,703 с на GPU T4/CUDA. CPU replay 129 validation-чипов (84 AF / 45 BS) дал Δ=0 во всех трёх checks. Независимый CSV validator завершился с EXIT 0 для 447 data rows: exact order, canonical RLE и BS exclusivity. Прошли public E2E (91 HTTP checks / 6 jobs), browser BS/export, rollback V3→V2 на BS159 с восстановлением V3 и parity raw mask CPU Windows/Linux на четырёх сценах. Это не организаторский speed score.
10. **Финальные gates.** Git хранит исходники и доказательства, а веса, данные и секреты остаются вне Git. Основной конкурсный репозиторий: [GitVerse](https://gitverse.ru/hackrus.experts/kosmo-krasnoiarsk_4_th_try_106). Зеркало разработки: [GitHub](https://github.com/foxxed20-ux/firewatch).

## Источники

* «Постановка кейса — Мониторинг природных пожаров КосмоХакатон 2026», с. 3, 9–15. Ссылка и SHA-256 указаны в [MODEL_API.md](../../MODEL_API.md).
* «Критерии оценки — Мониторинг природных пожаров КосмоХакатон 2026», с. 3–6. Ссылка и SHA-256 указаны в [MODEL_API.md](../../MODEL_API.md).
* Внутренние источники репозитория: [README.md](../../README.md), [README_MODEL.md](../../README_MODEL.md), [architecture.md](../architecture.md), [evaluation.md](../evaluation.md), `artifacts/split.json`.

## Обновление перед финалом
