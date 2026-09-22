# Phase 8C-A6: Multi-Identity Dataset Assembly

## 1. Goal

Expand the dataset from 2 identities to exactly 5 distinct identities (including ID_001 and ID_002) in order to provide the structural foundation for identity-generalization experiments (Phase 8C-B), while maintaining strict processing pipelines and preventing arbitrary domain mismatch.

## 2. Acquired Identities

All new identities were acquired from verified Public Domain (CC0 / U.S. Federal Government) video materials, ensuring explicitly permitted AI training use.

### `ID_003` - Donald Trump Weekly Address
- **Source:** Wikimedia Commons
- **Rights:** Public Domain / CC0
- **Sequences:** 1 sequence (`seq_01`)
- **Duration:** 28.0 seconds
- **Frames:** 700 frames
- **FPS:** 25.0

### `ID_004` - Joe Biden Weekly Address
- **Source:** Wikimedia Commons
- **Rights:** Public Domain / CC0
- **Sequences:** 1 sequence (`seq_01`)
- **Duration:** 32.0 seconds
- **Frames:** 800 frames
- **FPS:** 25.0

### `ID_005` - George W. Bush Address
- **Source:** Wikimedia Commons
- **Rights:** Public Domain / CC0
- **Sequences:** 1 sequence (`seq_01`)
- **Duration:** 26.0 seconds
- **Frames:** 650 frames
- **FPS:** 25.0

*(Note: ID_001 consists of the original 6 synthetic actor sequences [1196 frames] and ID_002 is the Barack Obama Pilot [750 frames].)*

## 3. Ingestion Validation

Every new identity (`ID_003`, `ID_004`, `ID_005`) successfully passed the identical ingestion pipeline:
1. **Frames:** Native 25 FPS successfully extracted into JPEGs (1280x720).
2. **Audio:** Multiplexed properly to 16kHz mono.
3. **Face/Landmarks:** 478-point mesh detected natively across all frames.
4. **V4 Projection:** Rigidly mapped to the frozen 11D PCA basis of the V4 model without refitting.

**OOD (Out-of-Distribution) Checks:** As anticipated, all new identities generated significant OOD warnings (`[-3, 3]` threshold violations) because the frozen PCA manifold was only fitted to `ID_001`. This effectively documents our generalization baseline; the temporal and identity generalization components will address these distribution shifts in Phase 8C-B.

## 4. Dataset Balance

- **Total Unique Identities:** 5
- **Total Sequences:** 10
- **Total Usable Frames:** 4096 (1196 + 750 + 700 + 800 + 650)
- **Speech Duration:** ~163 seconds (~2.7 minutes)
- **Smallest Identity:** ID_005 (650 frames / 26.0s)
- **Largest Identity:** ID_001 (1196 frames / 47.8s)

The dataset is moderately balanced. ID_001 remains the largest identity, which is useful given that the current weights are biased toward it.

## 5. Identity Split (Provisional)

A structurally capable multi-identity split has been proposed to ensure non-overlapping training, validation, and testing distributions for the upcoming architecture updates.

- **Train Identities:** `ID_001`, `ID_003`, `ID_004`
- **Validation Identity:** `ID_005`
- **Test Identity:** `ID_002`
- **Leakage Protection:** Strict isolation enforced by `identity_split_generator.py`. Absolutely no sequence or frame-level overlap occurs between splits.

## 6. Actual Storage Totals

| Asset | Files | Total MB | Total GB | Largest File | >100MB | >500MB | >1GB |
| ----- | ----: | -------: | -------: | -----------: | -----: | -----: | ---: |
| Raw Video | 11 | ~75 MB | 0.07 GB | ~15 MB | 0 | 0 | 0 |
| Extracted Frames | 4096 | ~333 MB | 0.33 GB | < 1 MB | 0 | 0 | 0 |
| Audio | 10 | ~10 MB | 0.01 GB | < 2 MB | 0 | 0 | 0 |
| Landmarks | 10 | ~16 MB | 0.02 GB | 4 MB | 0 | 0 | 0 |
| Motion Maps | 10 | < 1 MB | < 0.01 GB | < 1 MB | 0 | 0 | 0 |
| `.pt` Tensors | 20 | 3723 MB | 3.72 GB | 3606 MB | 1 | 1 | 1 |

*Note: `dataset_v4_images.pt` (3.6GB) remains the singular bottleneck file requiring Git exclusion, effectively matching our previous estimates exactly.*

## 7. Next Steps

The `READY — MULTI-IDENTITY DATASET COMPLETE` gate is cleared.
The dataset is now fundamentally capable of evaluating Identity Generalization logic (Phase 8C-B), while adhering to strict rights clearances, identical mechanical pipeline rules, and preserving our dataset isolation.
