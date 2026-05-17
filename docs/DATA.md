# Datasets

## Used

| Dataset            | License                          | Use                            | Source                                                       |
| ------------------ | -------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| COYO-700M          | CC-BY-4.0 (attribution required) | Stage 1, ~50M subset           | https://github.com/kakaobrain/coyo-dataset                   |
| JourneyDB          | Research-only                    | Stage 3 mix (not redistributed)| https://journeydb.github.io/                                 |
| BAGEL@50-NFE renders | Self-generated, Apache-2.0     | Stage 2 teacher samples        | produced by `scripts/render_teacher.py`                      |
| GenEval train      | Permissive                       | Stage 3                        | https://github.com/djghosh13/geneval                         |
| T2I-CompBench train| MIT                              | Stage 3                        | https://github.com/Karine-Huang/T2I-CompBench                |

## §Safety — datasets intentionally **excluded**

- **LAION-5B / LAION-aesthetic v2**: temporarily withdrawn by LAION in late 2023 after
  Stanford's CSAM audit (Thiel, 2023). While later re-released as Re-LAION-5B with
  filtering, downstream research-OSS use carries reputational and ethical risk that we
  judge unacceptable for a v0.1 release. Excluded by policy.
- **Datasets requiring scraping that violates source ToS**: excluded.
- **Datasets without a documented filtering pass for CSAM**: excluded.

## Attribution

COYO-700M is distributed by Kakao Brain under CC-BY-4.0. Per the license, attribution is
included in `MODEL_CARD.md` (created in P8) and re-distribution of the URL list (not the
images) follows Kakao Brain's redistribution protocol.

## Provenance recording

Every training run writes a `data_provenance.json` next to its checkpoint listing the
exact dataset versions, filter thresholds, and the proportion of each split used.
