# Learned Motion Representation Comparison (V2 vs V3)

## Objective
To determine whether preserving local phonetic detail via a temporal context stack `[Batch, Seq, Context, 80]` (V3) materially outperforms frame-averaged audio features `[Batch, Seq, 80]` (V2) under an identical sequence-level train/val/test split.

## Configurations
*   **V2**: Target frame at $t$ receives the average of all sub-frame `[1, 80]` log-mel windows overlapping $t$.
*   **V3**: Target frame at $t$ receives a causal sliding context of size $C=16$ (representing ~170.6ms of PAST audio). The model processes this `[16, 80]` stack using a temporal `Conv1D` contextual encoder before feeding the sequence into the LSTM.
*   **Future Lookahead**: `0ms`. The context is strictly causal, evaluating audio ending precisely at the frame boundary, ensuring **zero additional streaming latency**.
*   **Data Split**: Identical (4 Train, 1 Val, 1 Test).

## Evaluation Metrics

### 1. Generalization (TEST SPLIT)
| Metric | V2 (Averaged) | V3 (Contextual) | Difference |
| :--- | :--- | :--- | :--- |
| **Total MSE** | **0.0031** | 0.0032 | V3 worse by +0.0001 |
| **Total MAE** | **0.0326** | 0.0332 | V3 worse by +0.0006 |
| **Velocity Error** | 0.0004 | **0.0003** | V3 better by -0.0001 |
| **Temporal Jitter** | 0.0085 | **0.0061** | V3 better by -0.0024 |

### 2. Validation Split Insights
| Metric | V2 (Averaged) | V3 (Contextual) | Difference |
| :--- | :--- | :--- | :--- |
| **Total MSE** | **0.0076** | 0.0082 | V3 worse |
| **Lip Pucker MAE**| 0.0624 | **0.0586** | V3 better |
| **Head Pitch MAE**| 0.0604 | **0.0552** | V3 better |

## Key Findings

### 1. The Jaw "0.0000" MSE Mystery Solved
Full-precision analysis reveals that `jaw_open` MSE is literally `0.0000278`. However, examining the target distribution shows that the `jaw_open` target maxes out at `0.0085` (Mean `0.0001`). The MediaPipe face tracker is barely registering jaw articulation on this specific avatar's highly synthetic source footage. Therefore, the network achieves near-zero error largely because the target variance is near-zero, *not* because it perfectly solved human jaw kinematics.

### 2. Test < Validation Mystery Solved
The V2 Test error (`0.0031`) was lower than Validation (`0.0076`). Analysis of the target standard deviations reveals the Test sequence (`synthesia-assistant`) is exceptionally static compared to Validation (`practice-feedback`). For example, Test `lip_pucker` std is `0.0306`, while Val `lip_pucker` std is `0.1619` (5x more dynamic). The Test sequence simply contains less complex motion, leading to a mathematically lower MSE.

### 3. V2 vs V3 Decision
*   **V3 does not substantially improve scalar prediction error (MSE/MAE).** The temporal convolutions over the 170ms audio context slightly overfit, leading to a marginally worse Test MSE.
*   **V3 does improve temporal stability.** V3 reduced Test Jitter from `0.0085` to `0.0061`.
*   **Conclusion**: Providing sub-frame phonetic detail does *not* automatically revolutionize 4D coordinate prediction when using a simplistic Conv1D encoder. To utilize high-resolution temporal context effectively, a more advanced architecture (like Cross-Attention or a Transformer) may be required.

## Next Action
Given that providing contextual phonetic detail (V3) did not immediately overcome the dataset's low-variance limitations, the bottleneck is now the **Target Representation** (4 arbitrary heuristic blendshapes) and **Dataset Volume/Diversity**, not just the Audio Features. Before expanding architecture size, we must decide whether to continue predicting 4 blendshapes or transition to directly predicting dense facial latents / 2D landmarks.
