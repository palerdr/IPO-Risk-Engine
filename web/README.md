# IPO Risk Research dashboard

You can replay held-out predictions from a frozen real-price study. You can also open the Anthropic valuation case study.

```sh
npm ci
npm run dev
```

You run type checks and tests with `npm run build`. Vite writes the static application to `dist/`; `npm run preview` serves it. You need no backend or credentials to view the bundled study.

`App.tsx` reads `data/research.json` and displays the historical replay and evaluation. `Valuation.tsx` uses the independent DCF functions under `lib/valuation/`. You can regenerate the research report from the repository root with `uv run ipo-research evaluate`.

You hide future prices until you select Reveal outcome. This is a presentation aid; the exported study contains the known historical outcomes. You do not run a model in the browser or claim a prospective live forecast.

Read the root [study guide](../docs/STUDY.md) for model definitions and measured results. The dark presentation follows the dashboard example in the [Ryze guide](https://www.get-ryze.ai/blog/build-ad-dashboards-with-claude-ai-guide).

The original financial tests remain in `tests/valuation.test.ts`. You retain the retired interface and its interaction tests under `legacy/web/` at the repository root.
