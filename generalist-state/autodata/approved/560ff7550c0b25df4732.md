# italia-corpus-script

A daily pipeline that fetches Italian legal document collections from the [Normattiva Open Data API](https://api.normattiva.it), converts each document from **Akoma Ntoso XML** to Markdown with YAML frontmatter, and pushes the results into the [italia-corpus](https://github.com/ahmeabd/italia-corpus) repository — one subfolder per collection.

The script runs daily on **AWS Batch**, triggered by **Amazon EventBridge**, using a Docker image stored in **Amazon ECR** (`eu-north-1`).

---

## How It Works

```
Normattiva API
   │
   ├─ GET /collections/collection-predefinite   → list of predefined collections
   │
   └─ GET /collections/download/collection-preconfezionata?nome=…&formato=AKN&formatoRichiesta=V
         │
         └─ ZIP archive of Akoma Ntoso XML files
               │
               ├─ parse AKN 3.0 XML  →  extract frontmatter + body
               ├─ render to Markdown with YAML header
               └─ commit & push to GitHub  →  italia-corpus/<collection name>/
```

### Pipeline steps

1. **Fetch collections** — calls `/collections/collection-predefinite` and deduplicates by name, preferring format `V` (vigente) › `O` (originale) › `M` (multivigente).
2. **Clone target repo** — shallow clone (`--depth=1`) of `italia-corpus`, done once for all collections.
3. **For each collection:**
  - Downloads the ZIP archive (streaming, 3 retries with exponential backoff, 30 s connect / 300 s read timeout).
  - Validates the ZIP header (`PK` magic bytes).
  - Extracts AKN XML files and converts each to a Markdown file (see Output Format below).
  - Copies the Markdown files into a subfolder inside the cloned repo named after the collection (spaces preserved, hyphens in API names become spaces).
  - Commits with message `YYYY-MM-DD - <collection name>` and pushes to the default branch.
4. **Cleanup** — temporary ZIP, extracted XML directory, and Markdown staging directory are deleted after each collection, even on error.

---

## Output Format

Each Akoma Ntoso XML document becomes a single `.md` file with a YAML frontmatter block followed by the document body rendered as Markdown:

```markdown
---
tipo: decreto legislativo
numero: 196
data: 2003-06-30
titolo: "Codice in materia di protezione dei dati personali"
urn: urn:nir:stato:decreto.legislativo:2003-06-30;196
codice_redazionale: 003G0218
vigente: true
---

## Art. 1 — Diritto alla protezione dei dati personali

...
```

Inline `ref` elements are rendered as Markdown hyperlinks. Structural elements (`article`, `section`, `chapter`, `part`, `division`, `title`, `subtitle`) map to ATX headings. Filenames are sanitized and truncated to 200 bytes; if two documents in the same collection produce the same stem, a numeric suffix (`_2`, `_3`, …) is appended.

---

## Architecture

```
src/italia_corpus/
├── __main__.py        CLI entry point and argument parsing
├── config.py          Environment variables, API endpoints, logging
├── normattiva.py      Normattiva API client (fetch + deduplicate collections)
├── akn.py             Akoma Ntoso XML → frontmatter + Markdown renderer
├── converter.py       Batch conversion of an AKN XML directory to Markdown files
├── filename.py        Filename sanitization, truncation, and collision avoidance
├── git_ops.py         Git subprocess wrapper (clone, commit, push; tokens redacted from logs)
├── github_client.py   GitHub authentication, token rotation, repo creation
└── pipeline.py        Main orchestration (download → extract → convert → push)
```

---

## Module Reference

### `config.py`

Loads all environment variables via `python-dotenv` and exposes them as module-level constants. Also defines the two Normattiva API endpoint URLs, the download timeout tuple `(30 s connect, 300 s read)`, the retry count and sleep duration, and initialises the shared `italia_corpus` logger. Every other module imports constants from here — nothing reads `os.getenv` directly outside this file.

---

### `normattiva.py`

Handles communication with the Normattiva Open Data API.

- `fetch_predefined_collections()` — GETs `/collections/collection-predefinite` and returns the raw JSON list.
- `merge_collections_by_name()` — deduplicates the list: when the same collection name appears with multiple format codes, it picks one according to the priority `V` (vigente) › `O` (originale) › `M` (multivigente). Insertion order is preserved.
- `collection_download_params()` — builds the query-parameter dict for the download endpoint from a merged collection record.

---

### `akn.py`

Parses Akoma Ntoso 3.0 XML and renders it to Markdown. Works entirely with the standard-library `xml.etree.ElementTree` — no third-party XML dependency.

- `extract_frontmatter()` — walks the `<preface>` and `<meta>` nodes to populate an `AknFrontmatter` dataclass (`tipo`, `numero`, `data`, `titolo`, `urn`, `codice_redazionale`, `vigente`). Vigency is inferred from the absence of `<eventRef type="repeal">` entries in the lifecycle block.
- `format_frontmatter()` — serialises the dataclass to a YAML `--- … ---` block. The `titolo` field is always double-quoted and its inner quotes/backslashes are escaped.
- `body_to_markdown()` — walks `<preamble>`, `<body>`, and `<conclusions>` in order, dispatching structural tags (`article`, `section`, `chapter`, …) to ATX headings and `<p>` text to paragraphs. Heading depth increments with nesting.
- `_render_inline()` — handles inline content; `<ref>` elements are forwarded to `refs.resolve_ref()` for link resolution.

---

### `refs.py`

Resolves `<ref href="…">` targets to Markdown links. Two resolution strategies:

1. **Internal (corpus-relative)** — if the href's URN is present in the `urn_index` built from the already-cloned repo, a relative filesystem path is computed via `os.path.relpath` and the result is percent-encoded (spaces → `%20`, parentheses, brackets, `#`, `"`, `<>`, `\`). The `#fragment` is dropped because internal headings are not stable across runs.
2. **External fallback** — if the URN is not in the index, a `https://www.normattiva.it/uri-res/N2Ls?<urn>` link is returned unchanged (including any fragment), so the reader can still follow the reference on Normattiva.

`href_to_urn()` maps both `urn:nir:` identifiers and `/akn/it/act/…` path-style hrefs to a canonical `urn:nir:` string for index lookup.

---

### `frontmatter.py`

Reads and indexes the YAML frontmatter already present in corpus Markdown files. Used to build the URN → relative-path map that `refs.py` needs for cross-document linking.

- `read_frontmatter()` — parses the `--- … ---` block at the top of a `.md` file with two lightweight regexes (handles quoted and unquoted values; raises `MissingFrontmatterError` or `ParseError` on malformed input).
- `build_urn_index()` — scans the entire cloned repo with `rglob("*.md")` once at startup, extracting the `urn` field from each file and mapping it to its repo-relative path.
- `update_urn_index_from_paths()` — incrementally updates the index after each collection is pushed, so that later collections can resolve references to documents written earlier in the same run.

---

### `converter.py`

Orchestrates the conversion of a single collection directory.

`convert_akn_dir_to_md()` walks all `.xml` files under the extracted ZIP directory in **natural sort order** (numeric path segments sorted as integers, so `art_2` comes before `art_10`). For each file it:

1. Pre-parses the XML to derive the output filename from the document title before full rendering, so the `source_repo_path` (needed by `refs.py` for self-referential links) is known before writing.
2. Calls `akn_xml_to_markdown()` with the `urn_index` and the computed `source_repo_path`.
3. Writes the resulting Markdown and immediately registers the new file's URN in the live `urn_index`, making it available to subsequent documents in the same collection.
4. Appends a numeric suffix (`_2`, `_3`, …) on filename collisions.

Returns the count of successfully converted files; files that raise any exception are skipped with a warning.

---

### `filename.py`

All filename and folder-name logic lives here, keeping it testable in isolation.

- `safe_repo_name()` — slugifies a collection name for use as a temporary filesystem directory during extraction (lowercased, non-alphanumeric chars replaced with hyphens, consecutive hyphens collapsed).
- `safe_filename()` — sanitises a document title for use as a Markdown filename (preserves spaces, letters, digits, `.`, `_`, `-`; strips path separators).
- `fit_md_basename()` — ensures the final stem fits within 200 UTF-8 bytes. If it needs to be truncated, a 12-character SHA-256 hex digest of the original is appended as `_<digest>` to preserve uniqueness.
- `collection_subdir_name()` — produces the human-readable subfolder name used inside the corpus repo. Hyphens from the API name are converted to spaces; path-illegal characters are removed. Also run through `fit_md_basename` for the same length guarantee.

---

### `git_ops.py`

Thin wrapper around `subprocess` git calls.

- `git()` — runs any git command, capturing stdout/stderr and logging them at DEBUG level with any token strings redacted.
- `sync_collection_to_clone()` — copies the Markdown staging directory into the correct subfolder of the already-cloned working tree, runs `git add -A`, checks `git status --porcelain` to skip the push if nothing changed, commits with the message `YYYY-MM-DD - <collection name>`, and pushes to `origin HEAD:<branch>`.

The repo is cloned once and reused across all collections; only one push per collection is made rather than one per file.

---

### `github_client.py`

Handles all GitHub API interaction via PyGithub.

- `primary_token()` — returns the first non-empty token found by scanning `GITHUB_TOKEN_1` through `GITHUB_TOKEN_20`, falling back to `GITHUB_TOKEN`.
- `github_client()` — constructs a `Github` instance with retries disabled (retries are managed manually in `pipeline.py`).
- `verify_github_session()` — calls `GET /user` at startup to confirm the token is valid and logs a warning if the authenticated login does not match `GITHUB_USERNAME`, which would cause `get_repo` to look in the wrong namespace.
- `get_or_create_repo()` — returns the target repository if it exists; creates it as a private, auto-initialised repo if it does not. Logs actionable hints for 401/403 errors.

---

### `pipeline.py`

Top-level orchestrator that wires everything together.

1. Calls `fetch_predefined_collections()` + `merge_collections_by_name()`.
2. Resolves or creates the target GitHub repository.
3. Shallow-clones the repo into a `tempfile.TemporaryDirectory` and builds the initial `urn_index` from the existing Markdown files.
4. Iterates over collections: downloads the ZIP (streaming, with retry/backoff), validates the `PK` magic bytes, extracts, converts, pushes, then updates the live `urn_index`.
5. Handles `ENOSPC` explicitly to give an actionable error message.
6. Sleeps a random 2–5 seconds between collections to avoid triggering abuse rate limits.
7. Cleans up all temporary files in `finally` blocks, even on error.

---

## Infrastructure Example For Daily Updates

(This example is the actual daily job updating the repository)


| Component         | Service            | Details                                    |
| ----------------- | ------------------ | ------------------------------------------ |
| Schedule          | Amazon EventBridge | Daily cron trigger                         |
| Compute           | AWS Batch          | Runs the container job                     |
| Image registry    | Amazon ECR         | `eu-north-1`, image `italia-corups:latest` |
| Target repository | GitHub             | `ahmeabd/italia-corpus`                    |


---

## Prerequisites

- Python 3.13+
- Git (available in `PATH`)
- A GitHub Personal Access Token (classic) with `repo` scope — or a fine-grained token with read/write access to contents of the target repository and permission to create repositories.

---

## Setup

1. Copy the environment template:
  ```bash
   cp .env.example .env
  ```
2. Edit `.env` and fill in your values (at minimum `GITHUB_TOKEN_1` and `GITHUB_USERNAME`).
3. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```

---

## Environment Variables


| Variable                 | Required | Default                               | Description                                                                                                                                                             |
| ------------------------ | -------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GITHUB_TOKEN_1`         | Yes      | —                                     | Primary GitHub PAT. Supports rotation: the script checks `GITHUB_TOKEN_1` … `GITHUB_TOKEN_20`, using the first one set. Falls back to `GITHUB_TOKEN` if none are found. |
| `GITHUB_TOKEN`           | No       | —                                     | Fallback token used when no numbered token is set.                                                                                                                      |
| `GITHUB_USERNAME`        | Yes      | —                                     | GitHub user that owns the target repository. Must match the token owner.                                                                                                |
| `GITHUB_TARGET_REPO`     | No       | `italia-corpus`                       | Repository name where all collections are pushed (created automatically if it does not exist).                                                                          |
| `ROOT_PATH`              | No       | —                                     | Working directory for temporary ZIP downloads and extraction.                                                                                                           |
| `EXTRACTION_BUFFER_PATH` | No       | —                                     | Fallback working directory if `ROOT_PATH` is not set. In the Docker image this defaults to `/data`.                                                                     |
| `GIT_AUTHOR_NAME`        | No       | `GITHUB_USERNAME`                     | Git commit author name.                                                                                                                                                 |
| `GIT_AUTHOR_EMAIL`       | No       | `<username>@users.noreply.github.com` | Git commit author email.                                                                                                                                                |
| `LOGLEVEL`               | No       | `INFO`                                | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                                                                                                             |


---

## Running Locally

```bash
# Working directory from .env (ROOT_PATH or EXTRACTION_BUFFER_PATH):
python -m italia_corpus

# Or pass the working directory explicitly:
python -m italia_corpus /tmp/workdir
```

---

## Docker

```bash
docker build -t italia-corpus .

# /data is the default ROOT_PATH inside the container
docker run --env-file .env italia-corpus
```

The image is built on `python:3.13-slim-bookworm` with `git` and `ca-certificates` added.

---

## Deploy to AWS ECR

```bash
chmod +x deploy.sh
./deploy.sh
```

This builds for `linux/amd64` and pushes to the ECR repository at `eu-north-1`. After pushing, update the AWS Batch job definition to pick up the new image.

---

## Contributing

This script is maintained by [@ahmeabd](https://github.com/ahmeabd) as part of the [italia-corpus](https://github.com/ahmeabd/italia-corpus) open-data project.

Contributions are welcome. If you find a bug, have an idea for an improvement, or want to add support for additional Normattiva endpoints, feel free to open an issue or submit a pull request. When contributing:

- Open an issue first for any non-trivial change so we can discuss the approach.
- Keep pull requests focused — one concern per PR.
- Make sure the script still runs end-to-end locally before submitting.

If you find this project useful and want to help grow the corpus, starring the repository and spreading the word is also appreciated.

---

## License

MIT