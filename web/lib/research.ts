import report from "../data/research.json";

export { report };
export type Prediction = (typeof report.predictions)[number];
export type ModelId = keyof Prediction["probabilities"];
export type ResearchModel = (typeof report.models)[number];
