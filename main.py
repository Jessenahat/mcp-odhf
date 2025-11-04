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

# --- MCP tools + SSE (safe) ---
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio, json
from fastapi import Request

# CORS for public connector
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

class EmptyArgs(BaseModel):
    pass

class SearchFacilitiesArgs(BaseModel):
    province: str | None = None
    facility_type: str | None = None

@app.post("/tools/list_fields")
def tool_list_fields(_: EmptyArgs):
    if df is None:
        raise HTTPException(status_code=400, detail="CSV not loaded")
    return {"ok": True, "tool": "list_fields", "data": list(df.columns)}

@app.post("/tools/search_facilities")
def tool_search_facilities(args: SearchFacilitiesArgs):
    if df is None:
        raise HTTPException(status_code=400, detail="CSV not loaded")

    if COL_PROVINCE is None or COL_TYPE is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Expected columns not found in CSV.",
                "have": list(df.columns),
                "need_any_of": {
                    "province": list(ALIAS_MAP["province"]),
                    "odhf_facility_type": list(ALIAS_MAP["odhf_facility_type"]),
                },
            },
        )

    try:
        filt = df
        if args.province:
            filt = filt[filt[COL_PROVINCE].astype(str).str.contains(args.province, case=False, na=False)]
        if args.facility_type:
            filt = filt[filt[COL_TYPE].astype(str).str.contains(args.facility_type, case=False, na=False)]

        if filt.empty:
            return {"ok": True, "tool": "search_facilities", "data": [], "note": "No results with those filters."}

        # optional: tidy subset of columns
        preferred_cols = ["Facility Name", "City", COL_PROVINCE, COL_TYPE, "Postal Code", "Latitude", "Longitude"]
        use_cols = [c for c in preferred_cols if c in filt.columns]
        if use_cols:
            filt = filt[use_cols]

        return {"ok": True, "tool": "search_facilities", "data": df_to_records_clean(filt.head(25))}
    except Exception as e:
        # return 400 with the actual error text instead of 500
        raise HTTPException(status_code=400, detail=f"search_facilities failed: {e}")

# Minimal SSE: exposes the tools to ChatGPT
TOOLS_MANIFEST = [
    {"name": "list_fields", "description": "List dataset columns", "args_schema": {}},
    {"name": "search_facilities", "description": "Filter by province and/or ODHF facility type",
     "args_schema": {"province":"string?","facility_type":"string?"}},
]

@app.get("/sse")
async def sse(request: Request):
    async def eventgen():
        # 1) Primeiro evento IMEDIATO (descoberta de ferramentas)
        payload = {"event": "list_tools", "data": TOOLS_MANIFEST}
        yield {"event": "message", "data": json.dumps(payload)}

        # 2) Keepalive periódico (evita timeout de proxy)
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(10)
            yield {"event": "ping", "data": "keepalive"}

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",   # evita buffer em proxies
    }
    return EventSourceResponse(
        eventgen(),
        media_type="text/event-stream",
        headers=headers
    )
