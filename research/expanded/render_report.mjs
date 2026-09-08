// You need an external Data Analytics plugin installation to regenerate HTML.
// You pass its directory as the first argument; the repo does not bundle its renderer.
// You use the canonical report renderer with a scoped scrollbar-width correction.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const pluginRoot = process.argv[2];
if (!pluginRoot) throw new Error("Supply the installed Data Analytics plugin directory.");
const base = resolve(pluginRoot, "skills/build-report/scripts");
const { buildPortableArtifact, readPackagedReaderRuntime } = await import(pathToFileURL(resolve(base, "build_portable_artifact.mjs")));
const { deliverPortableArtifact } = await import(pathToFileURL(resolve(base, "deliver_portable_artifact.mjs")));
const runtime = readPackagedReaderRuntime().html;
// The packaged header uses 100vw, which includes the desktop scrollbar width.
const correction = "<style>.dashboard-shell .analytics-top-bar{width:calc(100% + var(--ds-gutter) + var(--ds-gutter));margin-inline:calc(-1 * var(--ds-gutter))}</style>";
const headEnd = runtime.lastIndexOf("</head>");
if (headEnd < 0) throw new Error("The packaged reader has no head element.");
const runtimeHtml = runtime.slice(0, headEnd) + correction + runtime.slice(headEnd);
const result = await deliverPortableArtifact({
  inputPath: "research/expanded/report-artifact.json",
  outputPath: "docs/CHALLENGER_REPORT.html",
  readyTimeoutMs: 15000,
}, { build: (input, options) => buildPortableArtifact(input, { ...options, runtimeHtml }) });
console.log(JSON.stringify(result));
if (!result.ok) process.exitCode = 1;
