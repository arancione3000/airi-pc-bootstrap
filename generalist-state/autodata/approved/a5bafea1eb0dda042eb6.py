from collections import defaultdict

import requests

from .config import COLLECTIONS_URL, logger


def fetch_predefined_collections() -> list[dict]:
    """GET /collections/collection-predefinite."""
    response = requests.get(COLLECTIONS_URL)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise TypeError("L'endpoint delle collezioni deve restituire un array JSON")
    logger.info("Fetched %d predefined collections from Normattiva", len(data))
    return data


def pick_formato_richiesta(codes: set[str]) -> str:
    """V se disponibile; altrimenti O; altrimenti M; altrimenti V come fallback."""
    if "V" in codes:
        return "V"
    if "O" in codes:
        return "O"
    if "M" in codes:
        return "M"
    return "V"


def merge_collections_by_name(rows: list[dict]) -> list[dict]:
    """Una sola riga per nomeCollezione, con formatoRichiesta scelto (V > O > M)."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        nome = (row.get("nomeCollezione") or row.get("nome") or "").strip()
        if nome:
            by_name[nome].append(row)

    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        nome = (row.get("nomeCollezione") or row.get("nome") or "").strip()
        if nome and nome not in seen:
            seen.add(nome)
            order.append(nome)

    out: list[dict] = []
    for nome in order:
        group = by_name[nome]
        codes = {(r.get("formatoCollezione") or "").strip().upper() for r in group} & {
            "O",
            "M",
            "V",
        }
        chosen = pick_formato_richiesta(codes)
        pick = next(
            (
                r
                for r in group
                if (r.get("formatoCollezione") or "").strip().upper() == chosen
            ),
            group[0],
        )
        merged = dict(pick)
        merged["nomeCollezione"] = nome
        merged["formatoCollezione"] = chosen
        out.append(merged)
    return out


def collection_download_params(collection: dict) -> dict:
    """Build query parameters for the collection download endpoint."""
    fr = (
        collection.get("formatoCollezione") or collection.get("formatoRichiesta") or "V"
    )
    fr = str(fr).strip().upper()
    if fr not in ("O", "M", "V"):
        fr = "V"
    return {
        "nome": (
            collection.get("nomeCollezione") or collection.get("nome") or ""
        ).strip(),
        "formato": "AKN",
        "formatoRichiesta": fr,
    }
