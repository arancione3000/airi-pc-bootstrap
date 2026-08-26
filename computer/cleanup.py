from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

HOME = Path.home()
QUARANTINE = HOME / '.local' / 'share' / 'airi-quarantine'
MANIFEST_DIR = QUARANTINE / 'manifests'

SAFE_AGE_DAYS = 7
OLD_DOWNLOAD_DAYS = 30
DUP_MIN_BYTES = 10 * 1024 * 1024
MAX_DUP_FILES = 800
MAX_HASH_BYTES = 80 * 1024 * 1024 * 1024


def _now() -> float:
    return time.time()


def _size(path: Path) -> int:
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for p in path.rglob('*'):
                try:
                    if p.is_file() and not p.is_symlink():
                        total += p.stat().st_size
                except (PermissionError, FileNotFoundError, OSError):
                    continue
            return total
    except (PermissionError, FileNotFoundError, OSError):
        return 0
    return 0


def _age_days(path: Path) -> float:
    try:
        return max(0.0, (_now() - path.stat().st_mtime) / 86400.0)
    except (PermissionError, FileNotFoundError, OSError):
        return 9999.0


def _file_entry(path: Path, category: str, risk: str, reason: str, auto_safe: bool) -> dict[str, Any]:
    try:
        st = path.stat()
        return {
            'path': str(path),
            'name': path.name,
            'size_bytes': st.st_size,
            'size_mb': round(st.st_size / 1048576, 2),
            'age_days': round(_age_days(path), 1),
            'category': category,
            'risk': risk,
            'reason': reason,
            'auto_safe': auto_safe,
        }
    except (PermissionError, FileNotFoundError, OSError):
        return {}


def _scan_file_candidates(limit: int = 300) -> list[dict[str, Any]]:
    now = _now()
    out: list[dict[str, Any]] = []
    # Deliberately conservative: only user-owned paths and clearly disposable classes.
    trash = HOME / '.local' / 'share' / 'Trash' / 'files'
    if trash.exists():
        for p in trash.rglob('*'):
            if len(out) >= limit:
                break
            try:
                if p.is_file() and p.stat().st_uid == os.getuid():
                    out.append(_file_entry(p, 'trash', 'low', 'Already in the user trash; safe to empty.', True))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    cache = HOME / '.cache'
    if cache.exists():
        for p in cache.rglob('*'):
            if len(out) >= limit:
                break
            try:
                if p.is_file() and p.stat().st_uid == os.getuid() and (now - p.stat().st_mtime) >= SAFE_AGE_DAYS * 86400:
                    out.append(_file_entry(p, 'cache', 'low', f'User cache file older than {SAFE_AGE_DAYS} days.', True))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    tmp = Path('/tmp')
    if tmp.exists():
        for p in tmp.iterdir():
            if len(out) >= limit:
                break
            try:
                if p.name.startswith(('airi-', 'playwright-')):
                    if p.stat().st_uid == os.getuid() and (now - p.stat().st_mtime) >= SAFE_AGE_DAYS * 86400:
                        out.append(_file_entry(p, 'temp', 'low', 'Old temporary file created by the Airi/browser runtime.', True))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    user_logs = [HOME / '.local' / 'state', HOME / '.cache' / 'logs', HOME / 'airi' / 'logs']
    for root in user_logs:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if len(out) >= limit:
                break
            try:
                if p.is_file() and p.stat().st_uid == os.getuid() and p.suffix.lower() in {'.log', '.old', '.tmp'} and (now - p.stat().st_mtime) >= 14 * 86400:
                    out.append(_file_entry(p, 'logs', 'low', 'Old user/runtime log file.', True))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    downloads = HOME / 'Downloads'
    installer_ext = {'.deb', '.rpm', '.msi', '.exe', '.dmg', '.pkg', '.iso', '.zip', '.7z', '.tar', '.gz'}
    if downloads.exists():
        for p in downloads.iterdir():
            if len(out) >= limit:
                break
            try:
                if p.is_file() and p.suffix.lower() in installer_ext and p.stat().st_uid == os.getuid() and _age_days(p) >= OLD_DOWNLOAD_DAYS:
                    out.append(_file_entry(p, 'old_download', 'medium', f'Old installer/archive in Downloads (>{OLD_DOWNLOAD_DAYS} days).', False))
            except (PermissionError, FileNotFoundError, OSError):
                pass

    return [x for x in out if x]


def _top_dirs() -> list[dict[str, Any]]:
    roots = [
        HOME / 'Downloads', HOME / 'Desktop', HOME / 'Documents', HOME / '.cache',
        HOME / '.local' / 'share' / 'Trash', HOME / 'airi'
    ]
    rows = []
    for p in roots:
        if p.exists():
            s = _size(p)
            rows.append({'path': str(p), 'size_bytes': s, 'size_gb': round(s / (1024**3), 3)})
    return sorted(rows, key=lambda x: x['size_bytes'], reverse=True)


def _duplicates() -> list[dict[str, Any]]:
    roots = [HOME / 'Downloads', HOME / 'Desktop', HOME / 'Documents']
    candidates: list[Path] = []
    scanned = 0
    total_bytes = 0
    by_size: defaultdict[int, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if len(candidates) >= MAX_DUP_FILES or total_bytes >= MAX_HASH_BYTES:
                break
            try:
                if not p.is_file() or p.is_symlink():
                    continue
                st = p.stat()
                if st.st_uid != os.getuid() or st.st_size < DUP_MIN_BYTES:
                    continue
                by_size[st.st_size].append(p)
                candidates.append(p)
                total_bytes += st.st_size
                scanned += 1
            except (PermissionError, FileNotFoundError, OSError):
                continue
    groups = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        hashes: defaultdict[str, list[Path]] = defaultdict(list)
        for p in paths:
            try:
                h = hashlib.sha256()
                with p.open('rb') as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
                hashes[h.hexdigest()].append(p)
            except (PermissionError, FileNotFoundError, OSError):
                continue
        for digest, same in hashes.items():
            if len(same) < 2:
                continue
            keep = max(same, key=lambda x: (x.stat().st_mtime, str(x)))
            removable = [x for x in same if x != keep]
            groups.append({
                'sha256': digest,
                'size_bytes': size,
                'copies': len(same),
                'keep': str(keep),
                'duplicates': [str(x) for x in removable],
                'reclaimable_bytes': size * len(removable),
                'risk': 'medium',
                'auto_safe': False,
            })
    return sorted(groups, key=lambda x: x['reclaimable_bytes'], reverse=True)


def scan() -> dict[str, Any]:
    disk = shutil.disk_usage(HOME)
    candidates = _scan_file_candidates()
    candidates.sort(key=lambda x: x.get('size_bytes', 0), reverse=True)
    safe = [x for x in candidates if x.get('auto_safe')]
    medium = [x for x in candidates if not x.get('auto_safe')]
    duplicates = _duplicates()
    safe_bytes = sum(x['size_bytes'] for x in safe)
    medium_bytes = sum(x['size_bytes'] for x in medium)
    dup_bytes = sum(x['reclaimable_bytes'] for x in duplicates)
    return {
        'disk': {
            'path': str(HOME),
            'total_bytes': disk.total,
            'used_bytes': disk.used,
            'free_bytes': disk.free,
            'total_gb': round(disk.total / 1024**3, 3),
            'used_gb': round(disk.used / 1024**3, 3),
            'free_gb': round(disk.free / 1024**3, 3),
            'used_percent': round(disk.used / disk.total * 100, 2) if disk.total else 0,
        },
        'top_directories': _top_dirs(),
        'candidates': candidates[:100],
        'summary': {
            'safe_reclaimable_bytes': safe_bytes,
            'safe_reclaimable_gb': round(safe_bytes / 1024**3, 3),
            'review_reclaimable_bytes': medium_bytes + dup_bytes,
            'review_reclaimable_gb': round((medium_bytes + dup_bytes) / 1024**3, 3),
            'duplicate_groups': len(duplicates),
            'duplicate_reclaimable_gb': round(dup_bytes / 1024**3, 3),
        },
        'duplicates': duplicates[:50],
        'policy': {
            'auto_clean_categories': ['trash', 'cache', 'temp', 'logs'],
            'manual_review_categories': ['old_download', 'duplicates'],
            'does_not_touch_system_paths': True,
            'never_deletes_without_scan': True,
        },
    }


def cleanup_safe(max_bytes: int | None = None) -> dict[str, Any]:
    report = scan()
    candidates = [x for x in report['candidates'] if x.get('auto_safe')]
    if max_bytes is not None:
        candidates = [x for x in candidates if x.get('size_bytes', 0) <= max_bytes]
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    deleted = []
    skipped = []
    for item in candidates:
        p = Path(item['path'])
        try:
            if not p.exists() or not p.is_file():
                skipped.append({**item, 'skip': 'missing_or_not_file'})
                continue
            # Final safety check: never touch outside clearly allowed roots.
            allowed = [HOME / '.cache', HOME / '.local' / 'share' / 'Trash' / 'files', Path('/tmp'), HOME / '.local' / 'state', HOME / 'airi' / 'logs']
            if not any(p.is_relative_to(a) for a in allowed):
                skipped.append({**item, 'skip': 'outside_safe_root'})
                continue
            deleted.append(item)
            manifest.append(item)
            p.unlink()
        except (PermissionError, FileNotFoundError, OSError) as e:
            skipped.append({**item, 'skip': type(e).__name__})
    manifest_path = MANIFEST_DIR / f'{int(_now())}.json'
    manifest_path.write_text(json.dumps({'created_at': _now(), 'deleted': manifest, 'skipped': skipped}, indent=2), encoding='utf-8')
    return {
        'ok': True,
        'deleted_count': len(deleted),
        'deleted_bytes': sum(x.get('size_bytes', 0) for x in deleted),
        'deleted_gb': round(sum(x.get('size_bytes', 0) for x in deleted) / 1024**3, 3),
        'skipped_count': len(skipped),
        'manifest': str(manifest_path),
        'note': 'Only low-risk disposable files were automatically removed. Old downloads and duplicates remain for review.',
        'after_scan': scan(),
    }
