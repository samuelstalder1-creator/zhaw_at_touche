# Thesis-relevante Ergebnisse

Alle Werte stammen aus den vorhandenen Result-Artefakten unter `results/`.
Metriken beziehen sich auf die positive Klasse `1` und den Test-Split.
Konfusionsmatrix: TN / FP / FN / TP.

## Alle Modelle

- `setup104-gemini`
- `setup104-gemma`
- `setup104-qwen`: als **QwenResidualStack** bei Touché eingereicht
- `setup104-base`: als **BaseStack** bei Touché eingereicht
- `setup119-gemini`
- `setup119-gemma`
- `setup119-qwen`: als **QwenResidualOnly** bei Touché eingereicht
- `setup10`: als **CompactQueryClassifier** bei Touché eingereicht
- `setup10_1-qwen`
- `setup10_1-gemma`
- `setup10_1_gemini`

| Setup | Beschreibung | Resultat | Accuracy | Recall | Precision | F1 | TN | FP | FN | TP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `setup10_1-gemma` | Longformer, Query + Gemma-Neutral + Response | vorhanden | 0.998 | 0.999 | 0.996 | 0.997 | 4308 | 8 | 2 | 1902 |
| `setup10_1-qwen` | Longformer, Query + Qwen-Neutral + Response | vorhanden | 0.998 | 0.995 | 0.998 | 0.996 | 4312 | 4 | 10 | 1894 |
| `setup10_1_gemini` | Longformer, Query + Gemini-Neutral + Response | vorhanden | 0.998 | 0.995 | 0.998 | 0.996 | 4312 | 4 | 10 | 1894 |
| `setup10` | ALBERT Compact Query Classifier | vorhanden | 0.993 | 0.983 | 0.995 | 0.989 | 4306 | 10 | 32 | 1872 |
| `setup104-qwen` | Full embedding stack mit Qwen-Neutral | vorhanden | 0.791 | 0.678 | 0.653 | 0.665 | 3629 | 687 | 614 | 1290 |
| `setup119-qwen` | Qwen Residual Only | vorhanden | 0.788 | 0.636 | 0.659 | 0.647 | 3689 | 627 | 694 | 1210 |
| `setup104-gemini` | Full embedding stack mit Gemini-Neutral | vorhanden | 0.772 | 0.662 | 0.619 | 0.640 | 3539 | 777 | 643 | 1261 |
| `setup119-gemini` | Gemini Residual Only | vorhanden | 0.766 | 0.601 | 0.622 | 0.612 | 3621 | 695 | 759 | 1145 |
| `setup104-gemma` | Full embedding stack mit Gemma-Neutral | vorhanden | 0.756 | 0.670 | 0.589 | 0.627 | 3425 | 891 | 628 | 1276 |
| `setup104-base` | Response-only embedding baseline | vorhanden | 0.737 | 0.657 | 0.560 | 0.605 | 3335 | 981 | 654 | 1250 |
| `setup119-gemma` | Gemma Residual Only | vorhanden | 0.727 | 0.718 | 0.540 | 0.617 | 3152 | 1164 | 536 | 1368 |

## Eingereichte Modelle bei Touché Challenge

- BaseStack (**Setup104-base**)
- QwenResidualStack (**Setup104-qwen**)
- QwenResidualOnly (**Setup119-qwen**)
- CompactQueryClassifier (**Setup10**)

| Model | Setup | Accuracy | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| CompactQueryClassifier | `setup10` | 0.993 | 0.983 | 0.995 | 0.989 |
| QwenResidualStack | `setup104-qwen` | 0.791 | 0.678 | 0.653 | 0.665 |
| QwenResidualOnly | `setup119-qwen` | 0.788 | 0.636 | 0.659 | 0.647 |
| BaseStack | `setup104-base` | 0.737 | 0.657 | 0.560 | 0.605 |

## Hinweise

- Alle oben gelisteten Modelle haben aktuell Result-Artefakte unter `results/`.
