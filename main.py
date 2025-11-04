from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import PlainTextResponse
from sse_starlette.sse import EventSourceResponse
import json
import asyncio

app = FastAPI(title="ODHF MCP Server (Minimal Safe)")

CSV_FILE = Path("odhf_v1.1.csv")
df = None

# --- Load CSV in background on startup (not during import)
@app.on_event("startup")
def load_data():
    global df
    def load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        for enc in ("utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="cp1252", errors="replace", low_memory=False)
    df = load_csv_safely(CSV_FILE)

# --- Column aliasing helper ---
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
    global df
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

# --- JSON-safe conversion helper ---
def df_to_records_clean(frame: pd.DataFrame):
    safe = frame.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")

# --- Health/debug endpoint ---
@app.get("/", response_class=PlainTextResponse)
def root():
    global df
    rows = None if df is None else int(len(df))
    return f"ODHF MCP Server is running! csv_found={CSV_FILE.exists()} rows={rows}"

# --- List columns endpoint ---
@app.get("/list_fields")
def list_fields():
    global df
    if df is None:
        raise HTTPException(status_code=400, detail=f"CSV not found at {CSV_FILE.resolve()}")
    return {"columns": list(df.columns)}

# --- Search endpoint ---
@app.get("/search_facilities")
def search_facilities(
    province: str = Query(None, description="Province or territory (e.g., Quebec, QC)"),
    facility_type: str = Query(None, description="ODHF facility type (e.g., Hospitals)"),
):
    global df
    if df is None:
        raise HTTPException(status_code=400, detail=f"CSV not found at {CSV_FILE.resolve()}")

    col_province = find_col(ALIAS_MAP["province"])
    col_type = find_col(ALIAS_MAP["odhf_facility_type"])
    if col_province is None or col_type is None:
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
        filtered = filtered[filtered[col_province].astype(str).str.contains(province, case=False, na=False)]
    if facility_type:
        filtered = filtered[filtered[col_type].astype(str).str.contains(facility_type, case=False, na=False)]

    if filtered.empty:
        return {"message": "No results. Try another province (e.g., 'QC'/'Quebec') or facility_type."}

    preferred_cols = [
        "Facility Name", "City", col_province, col_type,
        "Postal Code", "Latitude", "Longitude"
    ]
    subset = [c for c in preferred_cols if c in filtered.columns]
    if subset:
        filtered = filtered[subset]

    return df_to_records_clean(filtered.head(25))

# --- MCP manifest + one-shot SSE for ChatGPT (timout fix) ---
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
    async def gen():
        payload = {
            "event": "list_tools",
            "data": {"tools": TOOLS_MANIFEST},
        }
        yield f"event: message\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.05)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "text/event-stream; charset=utf-8",
    }
    return EventSourceResponse(gen(), media_type="text/event-stream", headers=headers)
