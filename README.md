# VeriSpectra

### *Single-Image Synthetic Face Detection via Hierarchical Anomaly Modeling*

------------------------------------------------------------------------

## 🚀 Overview

**VeriSpectra** is a hybrid anomaly-based deepfake detection framework
designed specifically for **single static images**.\
Unlike conventional deepfake detectors, VeriSpectra:

-   ❌ Does **not** require paired real--fake data
-   ❌ Does **not** rely on video frames
-   ❌ Does **not** need synthetic samples for training

Instead, it models the underlying **true face distribution** using a
hierarchical Gaussian Mixture Model (GMM) and multi-domain forensic
features.
A test image is classified by measuring how strongly it deviates from
the real-face manifold.

------------------------------------------------------------------------

## Key Capabilities
- Single-image deepfake detection
- Hybrid multi-domain feature extraction:
  - Residual noise features
  - Frequency-domain FFT features
  - Multi-radius LBP texture descriptors
  - CNN (EfficientNet-B0) visual embeddings
- Hierarchical GMM classifier
- Logistic Regression fusion layer for stable scoring

------------------------------------------------------------------------

## 🏗 Complete Architecture

![Architecture](.github/assets/Screenshot%202025-11-16%20133712.png)


------------------------------------------------------------------------


## Project Structure
```
VeriSpectra/
│── core/
│   ├── feature_extractors/
│   ├── gmm/
│   ├── train_gmm_vXX.py
│   ├── eval_gmm_vXX.py
│
│── models/
│── cache/
│── results/
│── Sample_img/
│── requirements.txt
│── README.md
```

------------------------------------------------------------------------

## Installation
```bash
git clone https://github.com/Soumyakanta-Sahoo/VeriSpectra.git
cd VeriSpectra

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Training (v42)
```bash
python core/train_gmm_v42.py
```

## Evaluation
```bash
python core/eval_gmm_v42.py
```

------------------------------------------------------------------------

## 📊 Performance Matrix

The Hierarchical GMM + Logistic Fusion model was evaluated across train, validation, and test splits.  
Performance remains highly consistent, demonstrating strong generalization without using synthetic data for training.

| Split | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|------|
| **Train** | **0.8511** | **0.8427** | **0.8633** | **0.8529** | **0.9262** |
| **Validation** | **0.8452** | **0.8186** | **0.8870** | **0.8514** | **0.9172** |
| **Test** | **0.8419** | **0.8114** | **0.8910** | **0.8493** | **0.9129** |

✔ High recall → strong synthetic face detection  
✔ Stable AUC ~0.91 → robust anomaly modeling  
✔ Balanced precision–recall → low error rates

------------------------------------------------------------------------

## Version Summary
| Version | Improvements |
|--------|--------------|
| v42    | Enhanced FFT bands, multi-radius LBP, hierarchical GMM, CNN fusion |


------------------------------------------------------------------------

## 📄 License
This project is released under the MIT License.  
See the **LICENSE** file for full text.

------------------------------------------------------------------------

## Citation
```
@software{VeriSpectra2025,
  author = {Sahoo, Soumyakanta and Singh, Anshika and Tiya},
  title  = {VeriSpectra: Hybrid Single-Image DeepFake Detection Framework},
  year   = {2025},
  url    = {https://github.com/Soumyakanta-Sahoo/VeriSpectra}
}

```

------------------------------------------------------------------------

## 👥 Authors

-   **Anshika Singh**
-   **Soumyakanta Sahoo**
-   **Tiya**

------------------------------------------------------------------------

## 📬 Contact

For collaborations or research discussion:\
📧 **acsoumyakanta@gmail.com**

------------------------------------------------------------------------
