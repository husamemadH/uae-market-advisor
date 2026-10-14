Milestone 11: executive evaluation

The twelve cases in cases.json exercise all eight advisory topics in Dubai and
Abu Dhabi, exact scenario arithmetic, a follow-up, and responsible uncertainty.
Each independent case starts a new SQLite session with its recorded profile.
E11 runs immediately after E01 in the same session. Selecting E11 also runs E01.
Natural-language scenario changes do not change the saved business profile.

Run from the project root after uv sync --locked:

  uv run python evaluate.py run --mode fixture
  uv run pytest -q

Fixture mode needs no key, makes no external requests, replaces the provider HTTP
boundary with scripted responses, and uses synthetic evidence from example.com.
Agno, the application service, tools, schema/reference checks and SQLite are real.
It checks the harness, not model judgment or advisory quality. Usage stays unknown
and the report refuses rubric scores for fixture runs.

Explicit paid evaluation, using OPENROUTER_API_KEY from the environment or .env:

  uv run python evaluate.py run --mode live --max-cost-usd 0.10
  uv run python evaluate.py run --mode live --max-cost-usd 0.03 --case E11

Live mode uses the application's selected model and current settings. Search is
enabled by default; ADVISOR_SEARCH_ENABLED=false or --no-search disables it and
the manifest records the choice. Search snippets remain uninspected leads.
--max-cost-usd must be finite and positive. This is a stop threshold between
turns, not a provider-enforced billing cap: the current turn can exceed it.
Missing cost stops subsequent paid turns. The runner does not retry cases. The
application now permits one bounded correction attempt for unretrieved evidence
references, using a fresh evidence registry and the remaining tool/search/output
budgets. Both attempt run IDs, the original validation error, and all observed
request usage are retained in the per-question record. SDK/agent retries stay
disabled. Saved baseline/revised full-suite artifacts predate this correction.
Use a provider-side spending limit if a hard billing cap is required.

Each execution creates a new runtime/evaluations/ directory (or --output PATH).
Existing output directories are refused, so reruns cannot replace failures.
Partial runs retain all twelve case entries with honest not_run reasons.
An unsuccessful prerequisite prevents its follow-up from running.

Artifacts:
  manifest.json: configuration, model/prompt/dataset versions, Python/package
    versions, Git commit, SHA256 of Python sources/cases/lockfile, execution order,
    timestamps, aggregate usage and stop reason. File hashes identify edits that
    were not yet committed at execution time. No environment/key dump is saved.
  E01.json ... E12.json: exact question and criteria, profiles before/after,
    session/run IDs, validated answer and rendered text, cited IDs, retrieved
    observations with dates/qualifiers, calculations, tool arguments/results,
    per-request usage, tool timing, latency, failures and automatic checks.
    If available, rejected_answer holds a structurally valid but rejected model
    answer for diagnosis only. It is never rendered as a successful answer.
  summary.json and report.txt: execution, rubric status, failures, usage and gate.
  sessions.db and runs.jsonl: local application persistence; not for publication.

Review and scoring:
  Inspect every material claim against the corresponding retrieved observation
  and its qualifiers. Open original sources when needed; source-ID validity alone
  does not establish support, freshness, city applicability or correct reasoning.
  Check assumptions and numerical statements even inside proposals/inferences.
  Check that each case actually supplies its requested decision, not just caveats.

  Score five dimensions 0, 1 or 2: relevance, evidence_support, actionability,
  business_reasoning, clarity. 0 means missing/materially wrong; 1 means mixed or
  incomplete; 2 means direct, supported or honestly qualified, and useful.
  Explicitly record hard failures such as invented citations, unsupported material
  facts, wrong arithmetic, hidden assumptions or session leakage. A score of 10
  cannot override a hard failure or an automatic contract failure.

  Save reviews.json as an array with one object per reviewed case:
    case_id: E01, etc.
    run_id: exact saved run_id (stale/duplicate reviews are rejected)
    reviewer: reviewer name
    reviewer_type: human or assistant_provisional
    scores: object containing the five dimension names with integer values 0–2
    hard_failures: array of descriptions, empty only when none found
    claim_checks: nonempty array identifying claims, records and support judgments
    notes: explanation of the score and expected criteria
    improvement: concrete next improvement, or explanation that none is needed

  uv run python evaluate.py report RUN_DIRECTORY --reviews RUN_DIRECTORY/reviews.json

The target is at least 10 of 12 cases scoring 8/10, all eight topics across both
cities, and no unresolved hard failures. The executive release gate additionally
requires human review of all twelve actual live cases. Assistant reviews are
provisional; they cannot pass that gate. No second model is called as a judge.
Coverage counts requested topic tags and answer city presence, not semantic
coverage. A human must assess substance. Other release work, including the demo
and final documentation, is outside this executive-evaluation gate.

The reviewed results folders contain selected, inspected artifacts from actual
executions. They preserve the baseline, subsequent reruns, review notes and the
evaluation report. Raw SQLite databases, environment files and unreviewed logs
stay under ignored runtime/. Answers use fixed public/synthetic case questions.
Do not publish private sessions when running your own questions.

Robustness coverage (all ordinary tests are offline):
  Missing key/startup: tests/test_chat.py.
  Authentication/credits/rate/timeout, malformed output, absent usage and storage:
    tests/test_usage.py (real SDK/Agno with mocked HTTP or SQLite faults).
  Search unavailable/empty/malformed/bounded, instruction-bearing snippets:
    tests/test_web_search.py and tests/test_search_integration.py.
    These establish data and validation boundaries, not model immunity to injection.
  Invented IDs and model-supplied URLs: tests/test_researched_answers.py.
  Commission 25 vs 0.25, nonfinite inputs, nonpositive contribution:
    tests/test_economics.py and tests/test_tool_integration.py.
  Different session budgets, reset and hypothetical inputs not persisted:
    tests/test_profiles.py, tests/test_sessions.py and tests/test_tool_integration.py.
  Historic/contradictory menu prices retain dates, portions/channels and review flags:
    tests/test_evidence.py and tests/test_evaluation.py.
  Harness isolation, E11 ordering, partial/budget-stop outcomes, review provenance,
    immutable rerun directories and hard-failure overrides: tests/test_evaluation.py.

Out-of-country requests, adversarial pressure to invent facts, and model handling
of conflicting sources require further behavioral evaluation. Existing validators
and fixtures do not prove that those model behaviors always work.
