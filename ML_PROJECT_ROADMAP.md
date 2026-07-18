# ML / Applied AI Project Roadmap

**Обновлено:** 2026-07-18  
**Главная цель:** через практические проекты выйти на уровень уверенного Applied AI / AI Engineer и собрать 1–3 кейса, которые можно показать работодателю.  
**Канонический файл проекта:** `D:\ML\My_first_model\ML_PROJECT_ROADMAP.md`.

Этот документ — единственный источник актуального плана. Подробные результаты отдельных экспериментов остаются в `reports/` и `experiment_ledger`, но порядок работ, статусы и принятые решения меняются только здесь.

## Правила проекта

1. Учиться через работающие эксперименты, а теорию разбирать на их примере.
2. Сначала делать минимальную рабочую версию, затем улучшать качество и архитектуру.
3. Не сравнивать модели «на глаз» без одинаковых seeds и формального evaluation-протокола.
4. Quick evaluation используется для итераций; full evaluation — только для финалистов.
5. Не заявлять результаты обучения, тестов, CUDA или производительности, если они реально не запускались.
6. Все важные операции записываются в experiment ledger: config, checkpoint, hash, команда, метрики, скорость, VRAM и вывод.
7. Каждый агент перед изменением направления читает этот roadmap; после согласованного изменения обновляет этот же файл.
8. RunPod и полноценный MLOps остаются отложенными, пока пользователь явно не вернёт их в активный план.

## Краткая последовательность

```text
CIFAR-10 DDPM
→ Tiny ImageNet U-Net
→ latent SiT-S/2 Imagenette
→ REPA и Evaluator
→ AFHQ Cats SiT-B/2
→ REPA early-stop
→ Portfolio Packaging & Repository Readiness
→ transfer Cat → all AFHQ
→ img2img
→ hires fix
→ Training/Evaluation Playbook и agent skills
→ RAE
→ native 256 px
→ Generated Image Inspector
→ CV navigation / robotics
```

## 1. Завершённые этапы

### 1.1 CIFAR-10 DDPM, U-Net, 32×32 — ✅ закрыт

- Реализована собственная class-conditioned DDPM на чистом PyTorch.
- Обучение завершено на 200k шагов на RTX 4090 с BF16.
- Исправлен детерминированный sampling, black/white failures устранены.
- Проверены raw, EMA, checkpoint/resume и фиксированные previews.
- Добавлен DDIM-50 для быстрых previews; DDPM-1000 сохранён для основной оценки.
- Оптимизация обучения дала примерно `+27.15%` throughput: `1787.57 → 2272.92 img/s`.
- Результат этапа: понимание DDPM, U-Net, noise schedule, loss, optimizer, EMA, sampling и checkpointing.

### 1.2 Tiny ImageNet U-Net, 64×64 — ✅ учебный этап завершён частично

- Создана более крупная U-Net примерно на 48.4M параметров.
- Обучение было остановлено примерно на 37%; checkpoint сохранён.
- Этап дал опыт перехода с 32×32 на 64×64, настройки batch/gradient accumulation и оценки производительности.
- Возвращаться к полному обучению сейчас не требуется; можно возобновить позже только при отдельной причине.

### 1.3 Latent SiT-S/2, Imagenette, 128×128 — ✅ закрыт как исследовательский этап

- Использован pretrained SD VAE и cached latents.
- Реализована SiT-S/2 с rectified/flow training и Heun sampling.
- Обучены baseline и отдельная REPA-версия с frozen DINOv2 teacher.
- REPA-run остановлен и сохранён примерно на 365k.
- Проверены raw/EMA, checkpoints, deterministic sampling и скорость.
- Результат этапа: переход от pixel diffusion/U-Net к latent flow/Transformer.

### 1.4 Training Evaluator — ✅ рабочая версия

- Зафиксированы quick и full протоколы.
- Метрики: FID, KID, precision, recall; для многоклассовых моделей также target accuracy.
- Диагностика: NaN/Inf, black/white/low-detail failures, feature duplicates, nearest neighbours и outliers.
- Используются одинаковые seeds, sampler, CFG, VAE, reference split и feature extractor.
- Реализовано сравнение нескольких checkpoints одной командой с JSON/CSV/Markdown и comparison grids.
- Небольшой технический долг: убедиться, что отчёт всегда выводит численную секцию `raw vs EMA`, а не только упоминает её.

### 1.5 AFHQ Cats SiT-B/2, 128×128, baseline — ✅ завершён

- Один класс, без REPA, чтобы быстро проверить способность более широкой модели учить качество и детали.
- Canonical checkpoint: raw 20k.
- Quick-200: FID `48.051`, KID `0.02052`, precision `0.340`, recall `0.754`.
- Raw лучше EMA; checkpoint заморожен как baseline.

### 1.6 AFHQ Cats SiT-B/2 + REPA, 128×128 — ✅ обучение и quick-сравнение завершены

- Отдельное обучение с нуля до 20k.
- REPA 10k → 20k продолжала улучшаться: FID `62.305 → 52.384`, precision `0.210 → 0.310`.
- Одновременно recall снизился `0.862 → 0.722`.
- На одинаковых 20k baseline лучше REPA по FID, KID, precision и recall.
- REPA EMA 20k сильно отстаёт; основной результат — raw.
- Вывод: REPA помогает учить структуру, но текущая always-on конфигурация не превзошла baseline на AFHQ Cats.

## 2. Текущий этап

### 2.1 Portfolio Packaging & Repository Readiness — ✅ завершён

**Решение:** packaging closeout принят. Подготовлены executive `README.md`, `docs/portfolio_case_study.md`, `docs/technical_retrospective.md`, четыре компактных visual assets, claim-to-evidence matrix, public verifier, third-party attribution и независимый Head-of-AI review. Репозиторий получил [MIT License](LICENSE), а публичное имя изменено на `Grigoriy-V/human-in-the-loop-generative-ml-lab`. Временный packaging gate снят; следующий готовый ML-шаг — `Cats → all AFHQ transfer`, но эта запись не авторизует его запуск без отдельного human-gated решения.

**Позиционирование:** один интегрированный публичный кейс — **Human-in-the-Loop Generative ML Lab**. Он показывает связную систему из двух ясно разделённых, но связанных частей:

1. практический путь generative ML: DDPM → latent SiT → evaluator → REPA/early-stop и честное сравнение;
2. контролируемый human-supervised multi-agent workflow: supervisor задаёт границы и решения, worker выполняет ограниченную работу, long runs остаются ручными, а evidence и решения фиксируются в ledger.

Это не два несвязанных проекта: orchestration обслуживает воспроизводимое исследование модели. При этом не делать завышенных production/MLOps заявлений: текущий процесс локальный, semi-automatic и human-gated.

#### Deliverables: три уровня рассказа

1. `README.md` — executive landing, 2–3 минуты: проблема, что построено и принято пользователем, измеримые результаты/ограничения, ссылка на доказательства и воспроизводимый вход.
2. `docs/portfolio_case_study.md` — интегрированный короткий кейс, 5–7 минут: отдельные секции **ML outcomes** и **agent orchestration / pipeline**, объединённые общей задачей и результатом.
3. `docs/technical_retrospective.md` — читаемая техническая ретроспектива, 8–12 минут, не research paper: путь обучения, сбои и решения, evaluator, REPA early-stop, воспроизводимость и agent workflow без копирования сырой хронологии ledger.

`docs/agent_orchestration.md` остаётся подробной reference-документацией, а не вторым portfolio story.

#### Visual package

Цель: 4–6 осмысленных, читаемых на GitHub визуалов с caption и alt text; по возможности переиспользовать реальные артефакты, не cherry-pick-ить вводящие в заблуждение примеры.

- timeline прогрессии модели и экспериментов;
- компактный AFHQ comparison metric chart с видимым ограничением precision/recall;
- representative fixed-seed generation/comparison grid;
- diagram `human supervisor → worker → tests/eval → ledgers → decision`;
- опционально компактная evidence/audit или repo-architecture графика — только если она добавляет новую информацию.

#### Порядок работ

A. Собрать inventory доказательств и claims из roadmap, reports, обоих ledger и существующих visual assets.
B. Провести public-repo readiness audit: secrets, крупные tracked artifacts, dead links, reproducibility, environment/setup, license/data/checkpoint disclosure, generated-output policy и навигация.
C. Сформировать information architecture/story outline и claim-to-evidence matrix.
D. Выбрать или создать visuals.
E. Переписать `README.md` и короткий case study.
F. Написать technical retrospective.
G. Провести независимый review с позиции Head of AI: link/render/claim checks.
H. Принять packaging closeout decision, затем снять gate и вернуться к `Cats → all AFHQ transfer`.

**Фактический closeout:** пункты A–H завершены. Проверка `python tools/verify_public_repo.py`, Markdown links, agent-ledger schema и `git diff --check` выполнена в release closeout; ML training/evaluation не запускались.

#### Критерии приёмки packaging closeout

- Внешний читатель понимает проблему, что пользователь лично построил/решил, измеримые результаты, новизну orchestration, ограничения и следующие шаги без чтения reports.
- Каждый количественный claim трассируется в report или ledger; completed work отделён от planned work, а user decisions — от agent execution.
- Локальный semi-automatic human-gated характер обучения изложен честно; нет раздутых production/MLOps claims.
- README краткий; visuals читаемы; нет broken links, secrets или крупных артефактов; documented reproducible entry points существуют.
- Technical retrospective укладывается примерно в 8–12 минут чтения и не дублирует chronologically raw ledgers.
- Репозиторий проходит final public-readiness checklist.

### 2.2 REPA early-stop: 10k REPA → 20k без REPA — ✅ завершён и зафиксирован

**Цель:** проверить, можно ли взять раннюю структурную пользу REPA, а затем дать модели свободно улучшать текстуры и мелкие детали.

Порядок:

1. Взять точный REPA checkpoint `step_0010000.pt`.
2. Создать отдельную ветку/output directory.
3. Продолжить с 10k до 20k с `repa_weight = 0`.
4. Полностью пропустить DINO teacher и projector после отключения REPA.
5. Не сбрасывать SiT weights, optimizer, LR scheduler, EMA или RNG.
6. Сохранить отдельный raw/EMA checkpoint на 20k.
7. Запустить одинаковый quick-200 для:
   - baseline raw 20k;
   - REPA always-on raw 20k;
   - REPA 10k → OFF → raw 20k;
   - EMA только диагностически.

Решение: по утверждённому supervisor решению текущий этап закрыт на evidence quick-200. Победитель — raw early-stop 20k: `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/best_raw_0020000.pt`, SHA-256 `300b5600b86d1a35ebf2c27307e480070cceee113735b23ffca8e46316e57bd0`; это hash-identical immutable copy `step_0020000.pt`. Он лидирует по FID `45.787` и KID `0.01692`. Baseline остаётся лучше по precision `0.340` против `0.280` и recall `0.754` против `0.732`; это зафиксированное ограничение выбора. Always-on REPA исключён из дальнейшего отбора. Full-1000 сознательно пропущен по решению supervisor; EMA не выбиралась.

### 2.3 Зафиксировать финальную AFHQ Cats модель — ✅ завершено для текущего ML-этапа

- Canonical raw checkpoint: `outputs/afhq_cat_sit_b_128_repa_early_stop/checkpoints/best_raw_0020000.pt`.
- Зафиксированы SHA-256, config, quick-200 metrics и fixed-seed comparison grids в отчёте и experiment ledger.
- Ограничение: freeze основан на quick-200, а не на full-1000; baseline сохранил преимущество по precision/recall.

## 3. Ближайшие следующие этапы

### 3.1 Transfer learning: Cats → все классы AFHQ — ⏸ следующий ML-шаг после packaging closeout

**Цель:** проверить полезность дообучения собственной модели на новом распределении, а не всегда начинать с нуля.

- Взять лучший AFHQ Cats checkpoint.
- Расширить conditioning с одного класса до `cat / dog / wild`.
- Корректно перенести общие SiT weights и отдельно инициализировать новые class embeddings.
- Не называть это обычным resume: это новый transfer/fine-tuning experiment.
- Сравнить короткий transfer-run с обучением с нуля при одинаковом бюджете.
- Оценивать каждый класс отдельно и общую выборку.

Результат: практический кейс по transfer learning, checkpoint surgery и multi-class conditioning.

### 3.2 Img2img для собственной модели — ⏭

Первая версия не требует нового обучения:

1. Закодировать входное изображение существующим VAE.
2. Добавить шум согласно параметру strength / стартовому времени.
3. Запустить обратный flow с этого состояния.
4. Сохранить class conditioning, seed и metadata.
5. Сделать strength-grid и измерить баланс сохранения композиции и изменения изображения.

После базовой версии решить, нужен ли отдельный обучающий objective для editing/control.

### 3.3 Hires fix / двойной прогон — ⏭

**Цель:** получить изображение выше 128 px без немедленного обучения native-256 модели.

- Первый проход: генерация 128×128.
- Upscale до совместимого размера около 192–256 px.
- Второй img2img-проход с небольшим noise strength.
- Сравнить сохранение композиции, детализацию и новые артефакты.
- Размер `196` не фиксировать без проверки: SD VAE и patch grid требуют совместимой кратности; первым безопасным кандидатом считать 192 px.

### 3.4 Sampler ablation — ⏭ после рабочей img2img/hires версии

- Сравнить Heun и Euler на одинаковых seeds и количестве NFE.
- Не расширять sweep без необходимости.
- Выбрать быстрый preview sampler и отдельный sampler для финальной оценки.

## 4. Автоматизация процесса обучения

### 4.1 Training & Evaluation Playbook v1 — ⏭ после early-stop и transfer

Вывести из проведённых экспериментов единый алгоритм:

1. dataset audit и split;
2. VAE/cache validation;
3. one-batch overfit;
4. performance benchmark;
5. short training smoke;
6. checkpoint/resume validation;
7. fixed preview protocol;
8. quick evaluation;
9. decision gate;
10. full evaluation финалиста;
11. report/model card/ledger update.

### 4.2 Agent skills и pipeline — ⏭ после стабилизации Playbook

- Сделать reusable skills для подготовки датасета, запуска smoke, обучения, evaluation и сравнения.
- Агент получает config/experiment manifest, а не длинное ручное ТЗ.
- Одна команда создаёт samples, metrics, comparison grids, отчёт и ledger entry.
- Агент отвечает за интерпретацию trade-offs и выбор следующего эксперимента.
- Не автоматизировать нестабильный процесс раньше, чем он дважды прошёл вручную по Playbook.

## 5. Следующая архитектурная ветка

### 5.1 RAE, 128×128 — 🧪 утверждённый кандидат

- Попробовать Representation Autoencoder вместо обычного SD VAE.
- Сначала проверить reconstruction ceiling и структуру latent space.
- Затем адаптировать SiT к новым latent channels/resolution.
- Сравнить с текущим VAE при одинаковом dataset/model budget.
- Не начинать до закрытия AFHQ transfer, img2img и Playbook v1.

### 5.2 Native 256×256 — 🧪 после 128px и RAE

- Перейти к native 256 только после доказанного 128px-рецепта.
- Начать с короткого capacity/data/VRAM benchmark.
- Выбрать SiT-S или SiT-B по реальной скорости и качеству, а не заранее.
- При необходимости вернуться к облаку, но локальный запуск остаётся основным до отдельного решения.

## 6. Applied AI / CV кейсы

### 6.1 Generated Image Inspector — 🎯 главный следующий портфельный продукт

Это отдельная production-система, не замена Training Evaluator.

**Training Evaluator:** сравнивает модели и checkpoints на наборах изображений.  
**Generated Image Inspector:** проверяет конкретные generated images и batch в production.

V1 без собственного датасета:

- pretrained DINO/DINOv2 features;
- CLIP/SigLIP для соответствия условию;
- готовые IQA/aesthetic модели;
- простые detectors/statistical rules;
- VLM только как объясняющий/семантический слой;
- batch report с issue tags и confidence.

Первая версия должна оставаться простой. Сложный human-feedback pipeline, собственный detector и разметка 300–1000 генераций обсуждаются позже, только если V1 покажет недостаток готовых моделей.

### 6.2 Creative QA — 🧪 расширение Inspector

- Brand/style consistency.
- Anatomy/artifact checks.
- Reference consistency.
- Batch drift и regression testing.
- Feedback loop и приоритизация ручной проверки.

### 6.3 CV navigation / robotics — 🧪 более поздний кейс

Порядок:

1. 2D-машинка по данным датчиков.
2. Добавить управление по изображению.
3. Indoor simulator: визуальная навигация робота в помещении.
4. Позже — perception/navigation на реальных autonomous-driving datasets, включая доступные NVIDIA-данные.

Фокус должен быть на CV: depth, segmentation, obstacle detection, localization и path planning, а не только на reinforcement learning.

## 7. MLOps / RunPod — ⏸ отложено до явного решения пользователя

Перед первым действительно тяжёлым облачным обучением один раз настроить базовый pipeline:

- создание pod через API;
- bootstrap окружения;
- Git/config/data sync;
- запуск и мониторинг обучения;
- сохранение checkpoints и logs;
- автоматическое завершение pod;
- управление Codex/агентом с минимумом ручных действий.

Сейчас локальная RTX 4090 остаётся основной средой. Этот этап не начинать, пока пользователь явно не скажет вернуть RunPod/MLOps в активную работу.

## 8. Непрерывная теория на практике

Теория изучается параллельно, на коде текущего проекта:

- tensors, shapes, batches и dataloaders;
- modules, layers, parameters и forward pass;
- autograd, gradients и optimizer step;
- learning rate, scheduler, warmup и gradient accumulation;
- Conv2d, normalization, activation, residual blocks;
- attention, tokens, patch size и SiT/DiT;
- VAE, latent space и reconstruction ceiling;
- DDPM, rectified flow, velocity prediction и samplers;
- EMA и raw weights;
- overfitting, memorization и data leakage;
- FID/KID/precision/recall и ограничения метрик;
- transfer learning, distillation и teacher/student training;
- production Python, reusable architecture и experiment tracking.

## 9. Портфельные результаты

### Кейс 1. Generative Model Training Lab

Собственный путь DDPM → latent SiT → REPA/early-stop → transfer → RAE, с воспроизводимыми экспериментами и честными сравнениями.

### Кейс 2. Evaluation & Generated Image Inspector

Training evaluator плюс production-oriented QA/Inspector с автоматическими метриками, comparison reports и issue detection.

### Кейс 3. Visual Navigation — позже

2D/indoor visual navigation с переходом к более реалистичным CV-данным.

Для каждого кейса подготовить README, архитектурную схему, reproducible commands, metrics, визуальные результаты, ограничения и короткий рассказ о принятых инженерных решениях.

## 10. Точный порядок ближайших действий

1. ✅ Завершить early-stop `REPA 10k → OFF → 20k`, quick-200 comparison и freeze raw checkpoint; full-1000 сознательно не запускался.
2. ✅ Закрыть `Portfolio Packaging & Repository Readiness`: README, case study, retrospective, visuals, verifier, attribution, MIT, независимый review и переименование публичного репозитория.
3. Следующий готовый шаг: подготовить и представить на отдельное human-gated утверждение `Cats → all AFHQ transfer`; не запускать обучение или evaluation автоматически.
4. После утверждённого transfer — реализовать img2img.
5. Реализовать hires fix около 192–256 px.
6. Сформировать Training & Evaluation Playbook v1.
7. Превратить стабильные операции в agent skills/pipeline.
8. Перейти к RAE 128.
9. После этого решить вопрос native 256 и/или облака.
10. Собрать Generated Image Inspector как отдельный Applied AI продукт.

## Как обновлять этот файл

После каждого решения меняются только четыре вещи:

1. статус этапа;
2. фактический результат и ссылка на подробный report;
3. принятое решение;
4. следующий конкретный шаг.

Историю всех запусков сюда не копировать — она остаётся в experiment ledger. Этот файл хранит направление проекта, а не сырые логи.
