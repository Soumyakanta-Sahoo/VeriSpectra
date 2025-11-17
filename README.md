# VeriSpectra: Single-Image Synthetic Face Detection via Hierarchical Anomaly Modeling

VeriSpectra is a forward-focused research framework for **single-image deepfake detection**, designed around a hybrid multi-domain feature pipeline and a lightweight hierarchical Gaussian Mixture Model (GMM).

## Key Capabilities
- Single-image deepfake detection
- Hybrid multi-domain feature extraction:
  - Residual noise features
  - Frequency-domain FFT features
  - Multi-radius LBP texture descriptors
  - CNN (EfficientNet-B0) visual embeddings
- Hierarchical GMM classifier
- Logistic Regression fusion layer for stable scoring

## Project Structure
```
VeriSpectra/
│── core/
│   ├── feature_extractors/
│   ├── gmm/
│   ├── train_gmm_vXX.py
│   ├── eval_gmm_vXX.py
│
│── utils/
│── models/
│── cache/
│── results/
│── Sample_img/
│── requirements.txt
│── README.md
```

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

## Version Summary
| Version | Improvements |
|--------|--------------|
| v42    | Enhanced FFT bands, multi-radius LBP, hierarchical GMM, CNN fusion |

## License
MIT License

## Citation
```
@software{VeriSpectra2025,
  author = {Anshika Singh, Soumyakanta Sahoo, Tiya},
  title  = {VeriSpectra: Hybrid Single-Image DeepFake Detection Framework},
  year   = {2025},
  url    = {https://github.com/Soumyakanta-Sahoo/VeriSpectra}
}
```
