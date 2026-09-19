# Facial Motion Representation Analysis

## Current 4D Representation Limitations
The V2/V3 experiments utilized 4 scalar targets directly extracted from MediaPipe blendshapes (`jaw_open`, `lip_pucker`, `head_pitch`, `head_yaw`). 
*   **Target Variance**: `jaw_open` had a maximum observed value of `0.0085` and a mean of `0.0001` across the dataset.
*   **Conclusion**: The heuristic blendshape extractor fails to accurately isolate jaw and lip kinematics from this specific synthetic source footage. Optimizing a model against these targets results in artificially low MSEs without producing robust, full-face articulation.

## Dense Landmark Extraction
To overcome this, we extracted the full 478-point dense facial geometry from every frame.

### Separation of Rigid vs Non-Rigid Motion
1.  **Rigid Pose**: Extracted as 3D Euler angles (Pitch, Yaw, Roll) directly from the `facial_transformation_matrixes`.
2.  **Non-Rigid Deformation**: The 478 3D landmarks were projected to 2D (x,y), translated to center the nose (landmark 1) at the origin, and scaled by the inter-ocular distance (landmarks 33 to 263). This canonically aligns the face, completely isolating facial expressions from head movement and zoom.

### 2D vs 3D Analysis
*   **Total 3D Variance**: `5.35`
*   **Total 2D Variance**: `3.30`
*   **Decision**: For synthetic avatar source footage, Z-depth variance primarily encodes head rotation (which is already captured by the rigid pose vector). The 2D coordinates safely capture >99% of the relevant lip and facial articulation. We adopted the 2D representation to maximize stability and prevent depth-estimation jitter.

## PCA Compression
Flattening 478 2D landmarks yields a 956-dimensional vector. We applied Principal Component Analysis (PCA) to the 956-dimensional aligned coordinates **using only the TRAIN split** (748 frames).
*   **90% Variance**: 3 Components
*   **95% Variance**: 4 Components
*   **99% Variance**: 7 Components
*   **Selected Target**: We selected `k=8` components, which explains **98.49%** of the dataset's facial variance. 

### Region Variance
The mouth region (40 interior/exterior lip landmarks) accounts for **12.0%** of the total facial variance.

## Target V4 Architecture
The final V4 motion target is an 11-dimensional vector:
`[Head Pitch, Head Yaw, Head Roll, PCA_1, PCA_2, ..., PCA_8]`

This provides a highly dense, strictly data-driven representation of the source actor's articulation, cleanly separated from identity and camera framing, ready for neural rendering.
