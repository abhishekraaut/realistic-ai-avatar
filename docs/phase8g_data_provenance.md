# Phase 8G: Data Provenance & Licensing Gate

All sequences ingested in the 10-identity pilot were subjected to a strict licensing gate.

## Validated Sources
- **MEAD (Multi-view Emotional Audio-Visual Dataset):** Used for ID_006, ID_007. License: CC-BY-NC 4.0 (Note: Commercial usage requires license upgrade; safe for internal R&D piloting).
- **CREMA-D:** Used for ID_008. License: Open Data Commons Open Database License (ODbL). 
- **Internal Opt-in Capture:** Used for ID_009, ID_010. License: Full commercial waiver acquired.

## Quarantined Sources
- `SEQ_034` (YouTube public scrape): Flagged and dropped. "Standard YouTube License" does not grant broad AI training or redistributive rights. Marked `BLOCKED — LICENSE UNCLEAR`.

## Provenance Standard
All future scale-out (50+ identities) must attach a `.license.json` manifest to the root of the identity directory explicitly listing the URL, Creator, and License type. Any ambiguity results in a hard failure at the preprocessing ingestion step.
