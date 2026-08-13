# Parser Latency Benchmark (R3-7)

This benchmark compares the optional free-text LLM parser with lightweight deployment alternatives.

| Mode | n | Median latency (s) | p95 latency (s) | Valid rate | Mean field accuracy |
|---|---:|---:|---:|---:|---:|
| cached_structured_parse | 8 | 0.0000 | 0.0000 | 1.00 | 1.00 |
| keyword_schema_parser | 8 | 0.0000 | 0.0002 | 1.00 | 0.60 |
| llm_parser_historical | 8 | 7.2374 | 8.1677 | 1.00 | 0.89 |
| structured_form_bypass | 8 | 0.0000 | 0.0000 | 1.00 | 1.00 |

Interpretation: structured forms and cache hits remove LLM latency from the deployment path. The deterministic keyword/schema parser is a lightweight baseline for standard briefs, whereas arbitrary natural-language briefs should still use the audited LLM parser or human review.
