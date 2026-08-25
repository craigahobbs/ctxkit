# AGENTS.md

Notes for coding agents working in this repository.

## Project

`ctxkit` is a Python CLI that builds AI prompts by concatenating ordered "context items" (messages, files, directories, URL content, included configs), optionally calls a provider API (Claude / Gemini / GPT / Grok / Ollama), and can extract response-emitted files back to disk (whole files or unified diffs).

## python-build

This is a [python-build](https://github.com/craigahobbs/python-build#readme) package. Read the python-build skill before running tests, lint, coverage, or changing the Makefile: [`../python-build/SKILL.md`](../python-build/SKILL.md) if that file exists, otherwise [https://raw.githubusercontent.com/craigahobbs/python-build/main/SKILL.md](https://raw.githubusercontent.com/craigahobbs/python-build/main/SKILL.md).

Local Makefile overrides:

- `PYLINT_ARGS` — missing docstring checks disabled
- no `SPHINX_DOC` — `make doc` is a no-op

`make clean` also cleans `examples/app-agent/`.

## Architecture

Entry point `ctxkit.main:main` (`src/ctxkit/main.py`) is the only non-trivial top-level flow. It:

1. Prepends `CTXKIT_FLAGS` env var to argv (used for user defaults like `--api grok grok-4-fast-reasoning`).
2. Parses argparse args — prompt-item flags (`-c -m -i -t -f -d -v`) all share `dest='items'` via the custom `TypedItemAction`, preserving **command-line order** as a list of `(item_type, value)` tuples. Item order matters: it is the order they appear in the final prompt.
3. Translates that list into a `{'items': [...]}` config dict matching the `CtxKitConfig` schema (defined as Schema Markdown in `config.py:CTXKIT_SMD`, parsed with `schema_markdown`). The schema is the source of truth — `-g/--config-help` prints it, and `-c` configs are validated against it.
4. Delegates to `config.process_config` / `process_config_items` (generator) to render each item to a string, then either prints, writes to `-o`, or pipes to `output_api_call`.

Key cross-cutting behaviors in `config.py`:

- **Variables** (`{{name}}`) — expanded in messages, paths, and template-included text. The `variables` dict is mutated in order, so a `var` item must appear *before* the items that reference it.
- **`config` items recurse** into `process_config_items` with `root_dir` set to the included config's directory; relative paths in a nested config resolve against *that* config's location, not the cwd.
- **`is_url`** (regex `^[a-z]+:`) is the URL/path dispatcher used in many places; `fetch_text` is the unified file-or-URL reader.
- **`--diff` mode** prepends `lineno:` to every file line on the way in (`_add_line_numbers`) and swaps the system prompt to `DEFAULT_SYSTEM_DIFF` (instructs the model to reply with unified diffs). On extraction, `diff.apply_diff` reconstructs files.

API layer (`src/ctxkit/api/`):

- `api/__init__.py` registers providers in `API_PROVIDERS` (`claude`, `gemini`, `gpt`, `grok`, `ollama`). Each provider module exports a `*_chat` generator (yields response chunks for streaming) and a `*_list` function (returns available model names). To add a provider: implement those two functions and add an entry to `API_PROVIDERS`.
- `output_api_call` streams chunks to the output, accumulates them, then runs `_extract_files` if `--extract` was passed.
- `_extract_files` parses the streamed response with `_R_FILENAME_TAG` (regex matching `<path>\n...\n</path>` blocks at line start). Content equal to `ctxkit: delete` deletes the file. Otherwise it writes (or, with `--diff`, applies a unified diff against the existing file).
- The default system prompts (`DEFAULT_SYSTEM`, `DEFAULT_SYSTEM_DIFF`) are what teach the model the `<filename>...</filename>` extraction protocol — changing the regex or the prompts requires updating both in lockstep.

## Conventions

- Single-package layout under `src/ctxkit/`; tests live in `src/tests/` and are named `test_<module>.py` (api tests use `test_api_<provider>.py`).
- 100% branch coverage is enforced — new code paths need tests, and pragmas (`# pragma: no cover`, `# pragma: no branch`) are used sparingly only where coverage is genuinely unreachable.
- No docstring lint requirements at module/class/function level (disabled in the Makefile).
