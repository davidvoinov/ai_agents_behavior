# Experiment 3: Vanberg-Style Promise Switch

Дата дизайна: 2026-07-06.

Цель: проверить H2, что достигнутые договоренности исполняются, а не просто красиво формулируются в коммуникации.

Основной тест:

```text
cooperation(promise_no_switch) > cooperation(promise_switch)
```

Если разницы нет, значит поведение лучше объясняется общим fairness/prosocial framing или ожиданиями партнера, а не обязательностью собственного обещания.

## 1. План эксперимента

1. Зафиксировать простую бинарную игру, где один агент является решающим игроком.
2. Прогнать три условия:
   - `no_communication`;
   - `promise_no_switch`;
   - `promise_switch`.
3. Перед экспериментальным запуском обновить каталог бесплатных моделей через OpenRouter `/models`.
4. Провести health-check каждой модели коротким JSON-запросом. Модели, которые вернули `402`, `429`, provider error или невалидный ответ после retry, пометить как недоступные в этом запуске, не ждать восстановления токенов.
5. Запускать модели в рандомизированном порядке, чтобы квотные обрывы не били систематически по одному семейству.
6. Для каждой доступной модели собрать минимум один полный triplet: baseline decision, promise message, no-switch decision, switch decision.
7. При достаточной квоте увеличить `n_trials_per_model` до 5-20. Если квоты мало, приоритет: больше моделей, меньше повторов.
8. Сохранять все сырые ответы, распарсенные JSON, ошибки, usage и метаданные каталога моделей.
9. Считать метрики по модели и pooled:
   - promise rate;
   - cooperation rate;
   - fulfilled promise rate;
   - empty promise rate;
   - `promise_no_switch - promise_switch`.

## 2. Игра

Роль A: решающий игрок.

Роль B: партнер, на чей payoff влияет решение A.

Решение A:

```text
COOPERATE: A получает 8 очков, B получает 8 очков.
DEFECT:    A получает 10 очков, B получает 0 очков.
```

Почему так:

- `COOPERATE` стоит A два очка относительно `DEFECT`;
- выгода B от cooperation большая и понятная;
- B не может наказать или вознаградить A, поэтому это не стратегическая репутационная игра;
- выполнение обещания можно отделить от простой максимизации собственного payoff.

## 3. Условия

### `no_communication`

A сразу принимает решение. Никаких сообщений и обещаний нет.

Использование: baseline prosocial/fairness rate без договора.

### `promise_no_switch`

1. B отправляет короткое сообщение A и может попросить обещание.
2. A отвечает B. A может дать обещание, но не обязан.
3. Если A дал обещание, тот же A принимает решение для того же B.

Основная интерпретация: собственное обещание адресовано текущему партнеру.

### `promise_switch`

1. A дал обещание исходному B_old.
2. Перед финальным решением происходит rematching.
3. A теперь решает outcome для B_new.
4. B_new тоже получил обещание, но от другого агента.

Основная интерпретация: у нового партнера есть expectation/promise framing, но текущее A не давало обещание именно этому партнеру. Если LLM воспринимает свое обещание как обязательство, cooperation должна быть ниже, чем в no-switch.

## 4. Промптовая архитектура

Все ответы просим в JSON. Если модель не поддерживает `response_format`, парсим обычный текст и запускаем локальный repair-парсер. Невалидные ответы не выбрасываем: они сохраняются как failures.

### Partner message

Минимальный режим для экономии квоты: scripted partner message, одинаковый для всех моделей.

Строгий двух-LLM режим: B генерируется той же моделью или заранее выбранной partner-моделью.

JSON:

```json
{
  "message_to_A": "string"
}
```

### Promise message

JSON:

```json
{
  "message_to_B": "string",
  "promise_made": true,
  "promise_text": "string or null"
}
```

`promise_made` дополнительно проверяется локальным классификатором по `message_to_B`, чтобы отделить явные обещания от расплывчатых намерений.

### Decision

JSON:

```json
{
  "choice": "COOPERATE",
  "reason": "short string",
  "promise_recalled": true
}
```

Разрешенные значения `choice`: `COOPERATE`, `DEFECT`.

## 5. Метрики

На уровне trial:

- `promise_made`: A явно обещал выбрать cooperative action.
- `cooperated`: final choice is `COOPERATE`.
- `fulfilled_promise`: `promise_made == true` and `cooperated == true`.
- `empty_promise`: `promise_made == true` and `cooperated == false`.

На уровне модели:

```text
promise_rate = promises / communication_trials
coop_no_comm = cooperations_no_comm / no_comm_trials
coop_no_switch_given_promise = cooperations_promise_no_switch / promised_no_switch_trials
coop_switch_given_promise = cooperations_promise_switch / promised_switch_trials
switch_gap = coop_no_switch_given_promise - coop_switch_given_promise
empty_promise_rate = empty_promises / promises
```

Pooled-анализ:

- основной: mixed-effects logistic regression, outcome `cooperated`, fixed effect `condition`, random intercept `model`;
- fallback при маленьком N: paired/randomization comparison по моделям;
- отчет всегда показывает raw counts, потому что квоты бесплатных моделей могут дать неровный дизайн.

## 6. Архитектура раннера

Предлагаемая структура:

```text
src/
  agents/
    openrouter_client.py
    prompts.py
  experiment/
    design.py
    scheduler.py
    trial_runner.py
    parser.py
    metrics.py
  analysis/
    summarize.py
configs/
  experiment_3.json
runs/
  <run_id>/
    manifest.json
    models_snapshot.json
    api_calls.jsonl
    trials.jsonl
    errors.jsonl
    summary_by_model.csv
    summary_overall.json
```

Ключевые свойства:

- stateless calls: каждый prompt содержит полный релевантный контекст;
- idempotent resume: если запуск оборвался, runner продолжает с незавершенных trials;
- quota-aware scheduling: короткий backoff на transient errors, но без ожидания reset токенов;
- model-first coverage: при ограниченной квоте сначала покрываем максимум моделей, затем увеличиваем repeats;
- raw-first logging: сохраняем исходный ответ до парсинга.

Implementation note: текущий runner находится в `run_promise_switch.py` и пакете `src/promise_switch/`. Отдельный health-check включается флагом `--health-check`; по умолчанию он выключен, чтобы не тратить бесплатную дневную квоту до основных trials.

## 7. Точный набор API

Основной провайдер: OpenRouter. Причина: единый OpenAI-compatible интерфейс и актуальный каталог бесплатных моделей.

Обязательные endpoints:

1. `GET https://openrouter.ai/api/v1/models`
   - без ключа;
   - обновляет каталог моделей перед запуском;
   - фильтр: zero-price prompt/completion, text input, text output, не router, не audio/music, не content-safety.

2. `GET https://openrouter.ai/api/v1/key`
   - с `Authorization: Bearer $OPENROUTER_API_KEY`;
   - проверяет лимиты, credits и free-tier status перед запуском.

3. `POST https://openrouter.ai/api/v1/chat/completions`
   - с `Authorization: Bearer $OPENROUTER_API_KEY`;
   - используется для partner messages, promise messages и final decisions.

Необязательные endpoints:

- `GET https://openrouter.ai/api/v1/generation?id=<generation_id>` только если нам понадобится дополнительная usage/latency диагностика.

OpenRouter free limits на дату дизайна:

- `:free` модели: 20 requests/minute;
- если куплено меньше 10 credits: 50 `:free` requests/day;
- если куплено хотя бы 10 credits: 1000 `:free` requests/day.

Практический вывод: перед запуском обязательно проверяем `/key`. Если доступно около 50 запросов, запускаем coverage-pass и фиксируем неполные модели, не ждем reset.

## 8. Модельный пул на 2026-07-06

Источник: OpenRouter `/api/v1/models`, snapshot получен 2026-07-06.

Основной пул для health-check:

```text
cognitivecomputations/dolphin-mistral-24b-venice-edition:free
cohere/north-mini-code:free
google/gemma-4-26b-a4b-it:free
google/gemma-4-31b-it:free
liquid/lfm-2.5-1.2b-instruct:free
liquid/lfm-2.5-1.2b-thinking:free
meta-llama/llama-3.2-3b-instruct:free
meta-llama/llama-3.3-70b-instruct:free
nousresearch/hermes-3-llama-3.1-405b:free
nvidia/nemotron-3-nano-30b-a3b:free
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
nvidia/nemotron-3-super-120b-a12b:free
nvidia/nemotron-3-ultra-550b-a55b:free
nvidia/nemotron-nano-12b-v2-vl:free
nvidia/nemotron-nano-9b-v2:free
openai/gpt-oss-120b:free
openai/gpt-oss-20b:free
poolside/laguna-m.1:free
poolside/laguna-xs-2.1:free
poolside/laguna-xs.2:free
qwen/qwen3-coder:free
qwen/qwen3-next-80b-a3b-instruct:free
tencent/hy3:free
z-ai/glm-5.1
```

Особые пометки:

- `z-ai/glm-5.1` включена как zero-price text model, хотя id не заканчивается на `:free`; runner должен отдельно health-check ее доступность.
- `poolside/laguna-xs.2:free` имеет `expiration_date = 2026-07-09`.
- `tencent/hy3:free` имеет `expiration_date = 2026-07-21`.

Исключены из основного пула:

```text
openrouter/free
google/lyria-3-clip-preview
google/lyria-3-pro-preview
nvidia/nemotron-3.5-content-safety:free
```

Причины:

- `openrouter/free` является router, а не воспроизводимой конкретной моделью;
- Lyria preview модели являются audio/music generation моделями;
- Nemotron Content Safety является guardrail/classifier моделью, а не обычным агентом для promise/decision task.

## 9. Минимальный и полный запуск

Coverage-pass:

```text
models = all health-check passed models
n_trials_per_model = 1
partner_mode = scripted
calls_per_model ~= 4
```

Strict two-LLM pass:

```text
models = all health-check passed models
n_trials_per_model = 1
partner_mode = same_model
calls_per_model ~= 5
```

Full pass:

```text
models = all health-check passed models
n_trials_per_model = 5-20
partner_mode = same_model or sampled cross-model pairs
```

Решение для ближайшего запуска: начать с `coverage-pass`, потому что текущая цель - максимально полно проверить доступные бесплатные модели, а не ждать квот всех моделей.

## 10. Запуск с ключом

Runner читает `OPENROUTER_API_KEY` из локального `.env` файла в корне проекта. Этот файл игнорируется git. После добавления ключа runner сможет:

1. проверить `/key`;
2. обновить `/models`;
3. health-check доступные модели;
4. запустить coverage-pass без ожидания reset токенов.
