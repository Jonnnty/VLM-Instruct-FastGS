# VLM-Instruct-FastGS: Semantic Guidance for Complete Scene Reconstruction

## 📌 Overview
VLM-Instruct-FastGS (Vision-Language Model Guided 3D Gaussian Splatting) enhances 3D Gaussian Splatting by leveraging Vision-Language Models (VLMs) to intelligently guide the densification process. Under the same iteration budget, our method achieves more complete scene reconstruction through a three-phase semantic guidance strategy:

- **Phase 1: Accelerated Detail Formation** – Identifies regions that are beginning to show texture detail, accelerating the reconstruction of main subjects during early training.
- **Phase 2: Background Completion** – Detects main subject regions using VLM understanding, then inverts these masks to obtain background areas requiring enhancement, ensuring full scene coverage.
- **Phase 3: Novel View Refinement** – Analyzes renders from unseen viewpoints to identify inconsistent or under-reconstructed regions, further improving rendering quality across the entire scene.

This semantic-aware approach enables comprehensive scene reconstruction without requiring additional iterations or manual annotation.
