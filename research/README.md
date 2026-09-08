# Frozen research inputs

You use `universe.json` to define a convenience sample before generating predictions. You use `input.json` to reproduce the current study without network access.

You retain 84 listings from 94 registry entries. You can inspect ten exclusions in the input and report. The source data comes from Yahoo Finance's public chart endpoint. The snapshot contains adjusted closing prices and volume, with SPY as the calendar and market reference.

You can refresh the source with `uv run ipo-research fetch --refresh`. A refresh overwrites the normalized input and can change results if the vendor revises its history. Preserve the current file to reproduce the current study. The command caches normalized downloads and exact upstream responses under `raw/`.

The generated report records a SHA-256 hash of `input.json`. It also records the universe hash and source-code hashes. You can export the report from the dashboard.

This sample does not establish a complete IPO population or point-in-time vendor history. See [the study limits](../docs/STUDY.md).
