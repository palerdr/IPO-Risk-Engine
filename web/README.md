# Dashboard guide

You can replay the frozen MVP forecasts or inspect the separate Anthropic DCF scenario. You need no backend or credentials.

Run from the repository root:

```sh
npm --prefix web ci
npm --prefix web run dev
```

Open the URL that Vite prints. Run `npm --prefix web run build` for type checks and tests, followed by the production build. Use `npm --prefix web run preview` to serve `web/dist/`.

## Read the interface code

You manage replay state in `App.tsx` and pass chart data to `components/ResearchCharts.tsx`. You read the frozen data and its TypeScript types through `lib/research.ts`. Both views use `lib/download.ts` for JSON exports.

You calculate valuation scenarios with the functions under `lib/valuation/` and present them in `Valuation.tsx`. The valuation engine uses assumptions; it does not consume classifier probabilities.

## Understand the evidence

You hide future prices until you select Reveal outcome. This is a presentation aid; the exported study contains known historical outcomes. You do not run a model in the browser or offer a prospective forecast.

You can regenerate the MVP snapshot with `uv run ipo-research evaluate`. The expanded CatBoost and TabNet study has a [separate report](../docs/CHALLENGER_REPORT.html); it does not feed this dashboard. Read the [study guide](../docs/STUDY.md) for the MVP and the [decision audit](../research/decision/README.md) for the limits of the current targets.

You maintain the frozen Anthropic evidence in `data/anthropic.v1.json`. Its `asOf` date describes the reviewed snapshot. Update the source records before presenting it as a later snapshot. You can inspect the visual reference in the [Ryze guide](https://www.get-ryze.ai/blog/build-ad-dashboards-with-claude-ai-guide).

You retain the financial fixtures in `tests/valuation.test.ts`. You keep the retired interface and its tests under `legacy/web/` at the repository root.
