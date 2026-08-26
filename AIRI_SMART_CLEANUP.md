# Airi-PC Smart Cleanup

Airi-PC includes a conservative disk-cleanup engine.

## Reports
- Total, used, free disk space and usage percentage.
- Largest user-facing directories: Downloads, Desktop, Documents, cache, Trash and Airi-PC runtime.
- Clearly disposable candidates: old user cache files, old Airi/Playwright temp files, old user logs and files already in the user's Trash.
- Review-only candidates: old installers/archives in Downloads and duplicate files in user folders.

## Safety
Automatic cleanup removes only low-risk disposable files under approved user roots. System paths are never touched. Old downloads and duplicates are not deleted automatically.

Each cleanup run writes a manifest under `~/.local/share/airi-quarantine/manifests/`.

## Tools
MCP: `computer_cleanup_scan`, `computer_cleanup_safe`.
HTTP: `GET /cleanup/scan`, `POST /cleanup/clean-safe`.
CLI: `airi-control cleanup-scan`, `airi-control cleanup-safe`.

Agent workflow: `scan -> explain -> safe cleanup -> rescan -> report reclaimed space`.
