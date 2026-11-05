# Llama Q4_K_M: plausible distractor, cohorte común

Ambos modelos recibieron los mismos manifests y n. La elegibilidad depende de si cada modelo
produjo dos distractores erróneos distintos. Las tasas de adopción de abajo usan la intersección
exacta de casos puntuados; la elegibilidad se informa sobre la cohorte completa.

| Dataset | Entrada | Q4 elegible | GPT-OSS elegible | Cohorte comparada | Q4 plausible | GPT-OSS plausible | Q4 implausible | GPT-OSS implausible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| medqa | 106 | 76 (71.7%) | 106 (100.0%) | 76 | 0.8158 | 0.1316 | 0.7500 | 0.0000 |
| medmcqa | 87 | 38 (43.7%) | 87 (100.0%) | 38 | 0.8421 | 0.4474 | 0.7105 | 0.1316 |

Datos completos y digests de cohorte: experiments/model_metadata/q4_plausible_distractor_common_cohort.json.
