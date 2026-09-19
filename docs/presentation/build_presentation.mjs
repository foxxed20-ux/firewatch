import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.cwd();
const SKILL_DIR = process.env.FIREWATCH_PRESENTATIONS_SKILL_DIR;
if (!SKILL_DIR) {
  throw new Error(
    "Set FIREWATCH_PRESENTATIONS_SKILL_DIR to the installed Presentations skill directory. See docs/presentation/presentation.md.",
  );
}
const TMP_DIR = path.join(workspaceDir, "artifacts", "presentation-build");
const FINAL_PPTX = path.join(
  workspaceDir,
  "docs",
  "presentation",
  "FireWatch_КосмоХакатон_2026.pptx",
);
const {
  resolvePresentationFont,
  applyPresentationChartFont,
  finalizePresentation,
} = await import(
  pathToFileURL(
    path.join(SKILL_DIR, "container_tools", "artifact_tool_utils.mjs"),
  ).href
);

await fs.mkdir(TMP_DIR, { recursive: true });
const metrics = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "docs", "presentation", "metrics.json"),
    "utf8",
  ),
);
const banner = await fs.readFile(
  path.join(workspaceDir, "docs", "assets", "firewatch-banner.svg"),
);
const demoScreenshot = await fs.readFile(
  path.join(workspaceDir, "docs", "assets", "firewatch-bs-demo.png"),
);
const font = resolvePresentationFont();
const W = 1280,
  H = 720;
const C = {
  navy: "#071C29",
  sea: "#133E44",
  cream: "#F6F5ED",
  ink: "#16313D",
  mint: "#A8D6C7",
  orange: "#FFAE72",
  coral: "#FF754C",
  pale: "#E7EEE9",
  gray: "#5F7279",
};
const deck = Presentation.create({ slideSize: { width: W, height: H } });

function box(
  slide,
  text,
  x,
  y,
  w,
  h,
  {
    size = 22,
    color = C.ink,
    bold = false,
    fill = "none",
    line = "none",
    align = "left",
    geometry = "textbox",
    italic = false,
  } = {},
) {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line:
      line === "none"
        ? { fill: "none", width: 0 }
        : { style: "solid", fill: line, width: 1 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: size,
    bold,
    italic,
    color,
    alignment: align,
    autoFit: "shrinkText",
  };
  return shape;
}
function rect(slide, x, y, w, h, fill, { radius = false, line = "none" } = {}) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line:
      line === "none"
        ? { fill: "none", width: 0 }
        : { style: "solid", fill: line, width: 1 },
  });
}
function base(title, number, dark = false) {
  const s = deck.slides.add();
  s.background.fill = dark ? C.navy : C.cream;
  rect(s, 64, 50, 42, 5, dark ? C.orange : C.coral, { radius: true });
  box(s, title, 64, 76, 980, 50, {
    size: 34,
    color: dark ? C.cream : C.ink,
    bold: true,
  });
  box(
    s,
    `FIREWATCH  /  КОСМОХАКАТОН 2026  /  ${String(number).padStart(2, "0")}`,
    64,
    675,
    640,
    18,
    { size: 11, color: dark ? C.mint : C.gray, bold: true },
  );
  return s;
}
function note(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}
function miniLabel(slide, text, x, y, w, color = C.gray) {
  box(slide, text.toUpperCase(), x, y, w, 18, { size: 11, color, bold: true });
}
function metric(slide, value, label, x, y, w, accent = C.orange) {
  box(slide, value, x, y, w, 52, { size: 38, color: accent, bold: true });
  box(slide, label, x, y + 54, w, 42, { size: 16, color: C.cream });
}

// 1. Cover
{
  const s = deck.slides.add();
  s.background.fill = C.navy;
  s.images.add({
    blob: banner,
    contentType: "image/svg+xml",
    alt: "FireWatch visual identity",
    fit: "cover",
    position: { left: 0, top: 0, width: W, height: 360 },
  });
  box(
    s,
    "Мониторинг природных пожаров\nпо спутниковым данным",
    72,
    430,
    760,
    120,
    { size: 44, color: C.cream, bold: true },
  );
  box(
    s,
    "AF: активное горение  /  BS: контуры и степень поражения",
    75,
    576,
    770,
    28,
    { size: 19, color: C.mint },
  );
  box(s, "КосмоХакатон 2026 · команда 4_th_try", 75, 650, 460, 22, {
    size: 14,
    color: C.orange,
    bold: true,
  });
  note(
    s,
    "Открывающий слайд. Визуальный источник: docs/assets/firewatch-banner.svg, собственный asset репозитория.",
  );
}

// 2. Problem
{
  const s = base("Пожар нужно увидеть в двух измерениях", 2);
  box(
    s,
    "Термоточка показывает место наблюдения, но не раскрывает масштаб последствий. FireWatch разделяет две задачи и связывает их в одном сервисе.",
    64,
    145,
    1040,
    62,
    { size: 25, color: C.ink },
  );
  rect(s, 64, 265, 515, 250, "#E1ECE7", { radius: true });
  rect(s, 625, 265, 515, 250, "#F9E5D4", { radius: true });
  miniLabel(s, "AF / active fire", 96, 302, 250, C.sea);
  box(s, "Где горит сейчас", 96, 330, 410, 44, {
    size: 30,
    color: C.ink,
    bold: true,
  });
  box(
    s,
    "Бинарная маска активного природного горения по VIIRS и вспомогательным признакам.",
    96,
    392,
    420,
    70,
    { size: 20, color: C.ink },
  );
  miniLabel(s, "BS / burn severity", 657, 302, 260, C.coral);
  box(s, "Что пострадало после пожара", 657, 330, 430, 44, {
    size: 30,
    color: C.ink,
    bold: true,
  });
  box(
    s,
    "Контур гари и классы слабой, средней и сильной степени поражения по Sentinel-1/2.",
    657,
    392,
    420,
    70,
    { size: 20, color: C.ink },
  );
  note(
    s,
    "Источник: README.md, раздел «Задача → решение»; MODEL_API.md, с. 3, 9–15 постановки кейса по ссылке в документе.",
  );
}

// 3. Inputs
{
  const s = base("Данные и выходы моделей", 3, true);
  miniLabel(s, "AF", 64, 154, 200, C.orange);
  box(
    s,
    "VIIRS: 8 полос\nAUX: 5 признаков\n23 признака модели",
    64,
    184,
    330,
    150,
    { size: 25, color: C.cream, bold: true },
  );
  box(s, "Результат\nмаска 0 / 1", 64, 385, 300, 80, {
    size: 27,
    color: C.mint,
    bold: true,
  });
  rect(s, 442, 150, 2, 365, C.mint);
  miniLabel(s, "BS", 502, 154, 200, C.orange);
  box(
    s,
    "Sentinel-2 до / после: 20 полос\nSentinel-1 до / после: 4 полосы\nAUX: 3 признака\n55 признаков модели",
    502,
    184,
    570,
    155,
    { size: 25, color: C.cream, bold: true },
  );
  box(s, "Результат\nфон + классы 1 / 2 / 3", 502, 385, 380, 80, {
    size: 27,
    color: C.mint,
    bold: true,
  });
  box(
    s,
    "Читатель входных чипов проверяет имена каналов, типы, размеры и совпадение геосеток.",
    64,
    560,
    1010,
    34,
    { size: 20, color: C.mint },
  );
  note(
    s,
    "Источник: README_MODEL.md, раздел «Данные и протокол»; MODEL_API.md, с. 9–15 постановки кейса.",
  );
}

// 4. Solution
{
  const s = base("От запроса к карте и файлам для ГИС", 4);
  const xs = [64, 306, 548, 790, 1032];
  const labels = [
    "AOI и период",
    "Каталог сцен",
    "Инференс AF / BS",
    "Контуры и площади",
    "Карта и экспорт",
  ];
  const details = [
    "bbox, Polygon\nили MultiPolygon",
    "Только явно\nзарегистрированные сцены",
    "Единый bridge\nдля CLI и API",
    "GeoJSON, GeoTIFF\nи JSON",
    "Слои, статус\nи скачивание",
  ];
  for (let i = 0; i < xs.length; i++) {
    rect(s, xs[i], 260, 185, 170, i === 2 ? "#F8D9C0" : "#E1ECE7", {
      radius: true,
    });
    box(s, String(i + 1).padStart(2, "0"), xs[i] + 18, 280, 54, 34, {
      size: 23,
      color: i === 2 ? C.coral : C.sea,
      bold: true,
    });
    box(s, labels[i], xs[i] + 18, 327, 150, 42, {
      size: 20,
      color: C.ink,
      bold: true,
    });
    box(s, details[i], xs[i] + 18, 380, 150, 45, { size: 15, color: C.gray });
    if (i < xs.length - 1)
      box(s, "›", xs[i] + 195, 323, 36, 44, {
        size: 38,
        color: C.coral,
        bold: true,
        align: "center",
      });
  }
  box(
    s,
    "Одна логика инференса служит конкурсу и геосервису: никаких выдуманных координат для обезличенного test.",
    64,
    532,
    1055,
    40,
    { size: 21, color: C.ink, bold: true },
  );
  note(
    s,
    "Источник: docs/architecture.md, разделы «Два пути к одному инференсу» и «Запрос на анализ»; MODEL_API.md, с. 3–6 критериев оценки.",
  );
}

// 5. Architecture
{
  const s = base("Архитектура: общий runtime, разные точки входа", 5, true);
  const nodes = [
    ["Конкурсный CLI", 64, 190, 250],
    ["Веб-карта", 64, 390, 250],
    ["FastAPI + очередь", 410, 390, 260],
    ["competition.service_bridge", 410, 190, 350],
    ["AF / BS модели", 860, 190, 250],
    ["Артефакты: CSV, GeoJSON, GeoTIFF", 860, 390, 290],
  ];
  for (const [t, x, y, w] of nodes) {
    rect(s, x, y, w, 82, "#143E4A", { radius: true, line: "#4C7776" });
    box(s, t, x + 16, y + 20, w - 32, 42, {
      size: 19,
      color: C.cream,
      bold: true,
      align: "center",
    });
  }
  const arrows = [
    [314, 226, 86, 0],
    [314, 426, 86, 0],
    [540, 272, 0, 104],
    [760, 230, 90, 0],
    [980, 272, 0, 104],
    [670, 426, 180, 0],
  ];
  for (const [x, y, w, h] of arrows) {
    const a = s.shapes.add({
      geometry: "line",
      position: { left: x, top: y, width: w, height: h },
      fill: "none",
      line: { style: "solid", fill: C.orange, width: 3 },
    });
  }
  box(
    s,
    "Модельный bridge проверяет manifest и bundle до запуска. API выдаёт job_id и хранит статус в SQLite.",
    64,
    570,
    1030,
    35,
    { size: 20, color: C.mint },
  );
  note(
    s,
    "Источник: docs/architecture.md; README.md, «Архитектура»; MODEL_API.md, с. 3–6 критериев оценки.",
  );
}

// 6. Protocol
{
  const s = base("Протокол валидации защищает от утечки", 6);
  box(s, "644 открытых train-чипа", 64, 155, 350, 46, {
    size: 32,
    color: C.ink,
    bold: true,
  });
  box(s, "420 AF + 224 BS", 64, 204, 250, 28, { size: 20, color: C.gray });
  rect(s, 64, 280, 1020, 76, "#DCE9E3", { radius: true });
  rect(s, 64, 280, 815, 76, C.sea, { radius: true });
  box(s, "515 train", 90, 303, 180, 25, {
    size: 23,
    color: C.cream,
    bold: true,
  });
  box(s, "129 validation", 895, 303, 165, 25, {
    size: 19,
    color: C.ink,
    bold: true,
  });
  const rows = [
    ["AF", "336 train / 84 validation", "Изолированы геосетки"],
    ["BS", "179 train / 45 validation", "Изолированы события и геосетки"],
  ];
  const t = s.tables.add({
    rows: 3,
    columns: 3,
    left: 64,
    top: 420,
    width: 1020,
    height: 135,
    values: [["Задача", "Разбиение", "Защита от пересечения"], ...rows],
  });
  t.styleOptions = { headerRow: true, bandedRows: true };
  t.borders.assign({ style: "solid", fill: "#B9CBC4", width: 1 });
  for (let c = 0; c < 3; c++) {
    t.getCell(0, c).fill = C.navy;
    t.getCell(0, c).text.style = {
      typeface: font,
      fontSize: 16,
      bold: true,
      color: C.cream,
    };
  }
  t.cells
    .block({ row: 1, column: 0, rowCount: 2, columnCount: 3 })
    .assign({ textStyle: { typeface: font, fontSize: 16, color: C.ink } });
  box(
    s,
    "Координаты, даты, ID событий и test metadata не входят в признаки модели.",
    64,
    590,
    1010,
    34,
    { size: 20, color: C.sea, bold: true },
  );
  note(
    s,
    "Источник: README_MODEL.md, «Данные и протокол»; docs/evaluation.md, «Данные и отсутствие утечки»; artifacts/split.json.",
  );
}

// 7. Results
{
  const s = base("Промежуточные метрики на validation", 7, true);
  box(s, metrics.scope, 64, 145, 1100, 28, {
    size: 20,
    color: C.mint,
    italic: true,
  });
  metric(s, metrics.af_f1.toFixed(4), "F1 активного горения", 64, 222, 220);
  metric(s, metrics.bs_burn_iou.toFixed(4), "IoU площади гари", 320, 222, 220);
  metric(
    s,
    metrics.bs_severity_miou.toFixed(4),
    "mIoU классов тяжести",
    576,
    222,
    240,
  );
  rect(s, 880, 208, 250, 146, "#F8D9C0", { radius: true });
  box(s, metrics.score.toFixed(4), 900, 233, 210, 50, {
    size: 38,
    color: C.coral,
    bold: true,
    align: "center",
  });
  box(s, "итоговый validation Score", 900, 295, 210, 36, {
    size: 16,
    color: C.ink,
    bold: true,
    align: "center",
  });
  const chart = s.charts.add("bar", {
    position: { left: 64, top: 405, width: 600, height: 205 },
    categories: ["IoU гари", "mIoU тяжести"],
    series: [
      {
        name: "dNBR comparator",
        values: [metrics.dnbr_burn_iou, metrics.dnbr_severity_miou],
        fill: "#78969A",
      },
      {
        name: "Tuned LightGBM",
        values: [metrics.bs_burn_iou, metrics.bs_severity_miou],
        fill: C.orange,
      },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 50 },
    hasLegend: true,
    legend: {
      position: "bottom",
      textStyle: { typeface: font, fontSize: 13, fill: C.cream },
    },
    yAxis: {
      min: 0,
      max: 0.7,
      majorUnit: 0.1,
      textStyle: { typeface: font, fontSize: 12, fill: C.cream },
      majorGridlines: { style: "solid", fill: "#4C7776", width: 1 },
    },
    xAxis: { textStyle: { typeface: font, fontSize: 13, fill: C.cream } },
    dataLabels: {
      showValue: true,
      position: "outEnd",
      textStyle: { typeface: font, fontSize: 12, fill: C.cream, bold: true },
    },
    chartFill: C.navy,
    plotAreaFill: C.navy,
  });
  applyPresentationChartFont(chart, { fontFamily: font });
  box(
    s,
    `Прирост BS: +${metrics.bs_burn_gain.toFixed(4)} IoU гари; +${metrics.bs_severity_gain.toFixed(4)} mIoU тяжести.`,
    710,
    422,
    400,
    58,
    { size: 20, color: C.cream, bold: true },
  );
  box(s, metrics.smoke, 710, 512, 380, 32, {
    size: 18,
    color: C.mint,
    bold: true,
  });
  box(
    s,
    "dNBR comparator не является официальным baseline. CPU replay v2 прошёл на 129 validation-чипах; результаты не оценивают закрытый test.",
    64,
    625,
    1090,
    25,
    { size: 17, color: C.mint },
  );
  note(
    s,
    "Источник: artifacts/ensemble-v2/selected_recipes.json; artifacts/tree-v1/tuned_recipes.json; artifacts/cnn-v1/af/validation_report.json; artifacts/ensemble-v2/demo_smoke/report.json. dNBR comparator: burnIoU 0.3638147045, mIoUsev 0.3542668560, пороги 0.1/0.2/0.4. CPU replay v2 на 129 validation-чипах: PASS. Это validation, не закрытый test Score.",
  );
}

// 8. Demo
{
  const s = base("Сервис и демонстрационный сценарий", 8);
  box(
    s,
    "Пользователь выбирает область и период, запускает AF, BS или обе задачи, затем получает слои, площади и экспорт.",
    64,
    150,
    1030,
    60,
    { size: 24, color: C.ink },
  );
  const labels = [
    "Выбор AOI\nи периода",
    "Запуск\nзадания",
    "Проверка\nстатуса",
    "Карта и\nартефакты",
  ];
  for (let i = 0; i < 4; i++) {
    const y = 260 + i * 78;
    rect(s, 64, y, 270, 58, i === 1 ? "#F9E5D4" : "#E1ECE7", { radius: true });
    box(s, String(i + 1), 82, y + 13, 35, 28, {
      size: 20,
      color: i === 1 ? C.coral : C.sea,
      bold: true,
    });
    box(s, labels[i], 128, y + 11, 180, 35, {
      size: 18,
      color: C.ink,
      bold: true,
    });
  }
  s.images.add({
    blob: demoScreenshot,
    contentType: "image/png",
    alt: "Проверенный экран FireWatch с результатом BS",
    fit: "contain",
    position: { left: 390, top: 245, width: 735, height: 387 },
    geometry: "roundRect",
    borderRadius: "rounded-xl",
  });
  box(
    s,
    "Реальный UI: official-ensemble-v2, BS_tr_000159, 938,5 га, покрытие 25,4%. Train/validation-демо, не закрытый test.",
    390,
    638,
    740,
    28,
    { size: 14, color: C.gray },
  );
  box(
    s,
    "Граница корректности: «нет данных» не означает «нет пожара». Готовность модели сервис сообщает отдельно от доступности API.",
    64,
    590,
    290,
    64,
    { size: 18, color: C.sea, bold: true },
  );
  note(
    s,
    "Источник: docs/assets/firewatch-bs-demo.png, копия artifacts/frontend/public-bs-desktop.png. Проверенный публичный HTTPS capture: official-ensemble-v2, BS_tr_000159, 938,5 га, coverage 25,4%; severity 242,9 / 176,4 / 519,1 га. Public BS GeoJSON download: PASS. Демо на train/validation, не закрытый test. Также: README.md, «Демонстрация»; docs/architecture.md, «Запрос на анализ»; docs/evaluation.md.",
  );
}

// 9. Verification
{
  const s = base("Проверяемое решение, а не только карта", 9, true);
  const groups = [
    ["Контракт данных", "Каналы, классы, размер, геосетки"],
    ["Конкурсный CSV", "sample_submission, порядок, RLE round-trip"],
    ["Model bundle", "manifest, SHA-256, совместимость входов"],
    ["REST API", "валидация AOI, jobs, статусы, GeoJSON"],
    ["Frontend", "состояния, карта, экспорт, ошибки"],
  ];
  for (let i = 0; i < groups.length; i++) {
    const y = 160 + i * 88;
    rect(s, 64, y, 235, 56, "#143E4A", { radius: true, line: "#4C7776" });
    box(s, groups[i][0], 82, y + 15, 195, 24, {
      size: 18,
      color: C.cream,
      bold: true,
    });
    box(s, groups[i][1], 335, y + 11, 700, 34, { size: 20, color: C.cream });
  }
  box(
    s,
    "V2: CLI 447 строк за 540,3 с; CPU replay 129 validation-чипов: PASS. Это не организаторский speed score.",
    64,
    605,
    1070,
    50,
    { size: 19, color: C.mint, bold: true },
  );
  note(
    s,
    "Источник: подтверждённый полный offline CLI-прогон v2 владельца model runtime: 269 чипов, 447 строк, 540,3043 с, сеть заблокирована. CPU replay всех 129 validation-чипов: PASS. Также: README.md, «Почему решение можно проверить» и «Проверки»; docs/evaluation.md, таблица «Что подтверждает каждый вид проверки». Время относится к среде команды, не к организаторскому speed score.",
  );
}

// 10. Limits & repo
{
  const s = base("Честные ограничения и следующий шаг", 10);
  rect(s, 64, 155, 510, 350, "#E1ECE7", { radius: true });
  miniLabel(s, "Что уже в репозитории", 96, 190, 340, C.sea);
  box(
    s,
    "Исходники моделей и сервиса\nКонтрактные тесты\nЗафиксированный split\nИнструкции запуска\nАрхитектура и протокол оценки",
    96,
    225,
    410,
    205,
    { size: 23, color: C.ink, bold: true },
  );
  rect(s, 620, 155, 510, 350, "#F9E5D4", { radius: true });
  miniLabel(s, "Осталось завершить", 652, 190, 380, C.coral);
  box(
    s,
    "Сравнение BS-кандидатов\nИтоговая поставка",
    652,
    225,
    410,
    110,
    { size: 23, color: C.ink, bold: true },
  );
  box(
    s,
    "Публичный E2E, rollback и parity CPU Windows/Linux на 4 сценах: PASS.",
    652,
    360,
    410,
    60,
    { size: 18, color: C.ink, bold: true },
  );
  box(
    s,
    "Команда 4_th_try",
    64,
    532,
    330,
    20,
    { size: 15, color: C.coral, bold: true },
  );
  box(
    s,
    "Git сохраняет этапы разработки: исходники, документацию, CI и историю версий. Данные, веса и секреты остаются вне Git.",
    64,
    560,
    1070,
    58,
    { size: 22, color: C.sea, bold: true },
  );
  box(
    s,
    "Конкурсный репозиторий: GitVerse · FireWatch",
    64,
    625,
    720,
    19,
    { size: 15, color: C.gray, bold: true },
  );
  box(
    s,
    "Зеркало разработки: GitHub · https://github.com/foxxed20-ux/firewatch",
    64,
    648,
    930,
    19,
    { size: 15, color: C.gray, bold: true },
  );
  note(
    s,
    "Команда: 4_th_try. Источник: README.md, «Данные и границы применения»; docs/evaluation.md, «Готовность релизного кандидата»; CHANGELOG.md. Публичный E2E, rollback и parity CPU Windows/Linux на 4 сценах: PASS. Основной конкурсный репозиторий: https://gitverse.ru/hackrus.experts/kosmo-krasnoiarsk_4_th_try_106. Зеркало разработки: https://github.com/foxxed20-ux/firewatch.",
  );
}

const candidatePath = path.join(TMP_DIR, "candidate-firewatch.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [],
};
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable:
    "C:/Users/FOX/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools",
    "inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools",
    "inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: [
    "--expected-slide-size-emu",
    "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide",
    "6",
  ],
  fontPolicy: { basis: "design", families: [font] },
  materializeLiteralChartWorkbooks: true,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "firewatch-presentation.validation.json"),
});

for (let i = 0; i < deck.slides.count; i++) {
  const png = await deck.slides.getItem(i).export({ format: "png", scale: 1 });
  await fs.writeFile(
    path.join(TMP_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`),
    new Uint8Array(await png.arrayBuffer()),
  );
}
console.log(FINAL_PPTX);
