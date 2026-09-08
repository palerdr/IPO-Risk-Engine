SELECT stage, model,
       COUNT(*) AS test_cases,
       SUM(event) AS events,
       AVG((probability - event) * (probability - event)) AS brier,
       AVG(probability) AS mean_probability
FROM forecasts
GROUP BY stage, model
ORDER BY stage, brier;
