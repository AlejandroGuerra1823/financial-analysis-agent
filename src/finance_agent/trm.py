"""Optional enrichment: Colombia's official TRM (USD/COP) from datos.gov.co.

Same public dataset used by the companion project
https://github.com/AlejandroGuerra1823/colombia-finance-mcp — here we only need
the latest rate to express totals in USD. Failures are non-fatal: the report
simply omits USD figures.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

TRM_URL = "https://www.datos.gov.co/resource/32sa-8pi3.json"


def fetch_latest_trm(timeout_seconds: float = 5.0) -> float | None:
    query = urllib.parse.urlencode({"$order": "vigenciadesde DESC", "$limit": "1"})
    try:
        with urllib.request.urlopen(f"{TRM_URL}?{query}", timeout=timeout_seconds) as response:
            records = json.load(response)
        value = float(records[0]["valor"])
        return value if value > 0 else None
    except Exception:
        return None
