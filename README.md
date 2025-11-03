# MCP for Statistics Canada – ODHF

This project exposes an **MCP server** for the [Open Database of Healthcare Facilities (ODHF)](https://www.statcan.gc.ca/en/lode/databases/odhf).

## Features
- `list_fields` → list dataset column names.
- `search_facilities(province?, facility_type?)` → query up to 25 facilities.
- `/sse` → provides MCP tool manifest for ChatGPT integration.

## Run Locally
```bash
uvicorn main:app --reload --port 8888
