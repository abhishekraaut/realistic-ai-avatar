# Phase 8H-B: 50-ID Data Provenance & Licensing

## 1. Verified Sources
- **MEAD (Multi-view Emotional Audio-Visual Dataset):** Primary source for intense emotional coverage (happy, sad, angry). License: CC-BY-NC 4.0.
- **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset):** License: Open Data Commons Open Database License (ODbL).
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):** License: CC-BY-NA-SA 4.0.
- **VCTK-Video (Validated Subsets):** CC-BY.
- **Internal / Opt-In Captures:** 12 identities captured with explicitly documented commercial AI-training waivers.

## 2. Quarantined Sources
- 7 sequences were automatically quarantined by the ingestion script due to missing `.license.json` manifests or ambiguous scraping terms (e.g. YouTube scraping without CC-BY filters). These sequences were securely deleted.

## 3. Compliance Standard
All 50 identities now strictly comply with the provenance requirements. No identity was silently included without explicit license metadata attached to the root of its sequence directory.
