from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse

app = FastAPI(title="ODHF MCP Server (Minimal Safe)")

CSV_FILE = Path("odhf_v1.1.csv")   # adjust if you renamed your file

# ---------------- CSV loader (robust encodings) ----------------
def load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    # last resort: replace bad chars
    return pd.read_csv(path, encoding="cp1252", errors="replace", low_memory=False)

df = load_csv_safely(CSV_FILE)

# ---------------- column aliasing ----------------
ALIAS_MAP = {
    "province": {
        "province", "Province", "Province or Territory", "Province/Territory",
        "prov", "province_or_territory"
    },
    "odhf_facility_type": {
        "odhf_facility_type", "ODHF Facility Type", "Facility Type", "facility_type",
        "odhf facility type"
    },
}

def find_col(candidates: set[str]) -> Optional[str]:
    if df is None:
        return None
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for want in candidates:
        if want in cols:
            return want
        if want.lower() in lower:
            return lower[want.lower()]
    return None

COL_PROVINCE = find_col(ALIAS_MAP["province"])
COL_TYPE     = find_col(ALIAS_MAP["odhf_facility_type"])

# ---------------- JSON-safe conversion helper ----------------
def df_to_records_clean(frame: pd.DataFrame):
    """
    Convert a DataFrame to JSON-safe records:
    - replace NaN/NaT with None
    - replace ±inf with None
    """
    safe = frame.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")

# ---------------- simple health/debug ----------------
@app.get("/", response_class=PlainTextResponse)
def root():
    rows = None if df is None else int(len(df))
    return f"ODHF MCP Server is running! csv_found={CSV_FILE.exists()} rows={rows}"

@app.get("/list_fields")
def list_fields():
    if df is None:
        raise HTTPException(status_code=400, detail=f"CSV not found at {CSV_FILE.resolve()}")
    return {"columns": list(df.columns)}

# ---------------- search endpoint ----------------
@app.get("/search_facilities")
def search_facilities(
    province: str = Query(None, description="Province or territory (e.g., Quebec, QC)"),
    facility_type: str = Query(None, description="ODHF facility type (e.g., Hospitals)"),
):
    if df is None:
        raise HTTPException(status_code=400, detail=f"CSV not found at {CSV_FILE.resolve()}")

    if COL_PROVINCE is None or COL_TYPE is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Expected columns not found.",
                "have": list(df.columns),
                "need_any_of": {
                    "province": list(ALIAS_MAP["province"]),
                    "odhf_facility_type": list(ALIAS_MAP["odhf_facility_type"]),
                },
            },
        )

    filtered = df
    if province:
        filtered = filtered[filtered[COL_PROVINCE].astype(str).str.contains(province, case=False, na=False)]
    if facility_type:
        filtered = filtered[filtered[COL_TYPE].astype(str).str.contains(facility_type, case=False, na=False)]

    if filtered.empty:
        return {"message": "No results. Try another province (e.g., 'QC'/'Quebec') or facility_type."}

    # (optional) choose a tidy subset of columns if you like:
    preferred_cols = [
        "Facility Name", "City", COL_PROVINCE, COL_TYPE,
        "Postal Code", "Latitude", "Longitude"
    ]
    subset = [c for c in preferred_cols if c in filtered.columns]
    if subset:
        filtered = filtered[subset]

    return df_to_records_clean(filtered.head(25))

# --- MCP manifest + one-shot SSE for ChatGPT ---
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
import json, asyncio

TOOLS_MANIFEST = [
    {
        "name": "list_fields",
        "description": "List dataset columns",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_facilities",
        "description": "Search facilities by province and/or ODHF facility type",
        "input_schema": {
            "type": "object",
            "properties": {
                "province": {"type": "string"},
                "facility_type": {"type": "string"},
            },
        },
    },
]

@app.get("/sse_once")
async def sse_once(_: Request):
    """Emit a single MCP discovery event, then close (best for ChatGPT)."""
    async def gen():
        payload = {
            "event": "list_tools",
            "data": {"tools": TOOLS_MANIFEST},  # <-- required shape
        }
        # raw SSE: event + data + blank line
        yield f"event: message\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.05)  # allow flush, then close

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    }
    return EventSourceResponse(gen(), media_type="text/event-stream", headers=headers)
