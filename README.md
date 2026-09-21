# face-recognition-system
AI-powered Face Recognition Identification System built with Python, FastAPI, OpenCV, and dlib. Supports face enrollment, 128-D feature embeddings, similarity-based identification, unknown-face rejection, evaluation metrics, CLI tools, and an interactive glassmorphism web interface.


An enterprise-grade, full-stack **Face Recognition Identification System** built in Python with a **FastAPI REST API Backend**, an **attractive glassmorphism Web Frontend UI**, and robust **CLI tools**.

The system **enrolls** individuals from photo datasets, **identifies** query faces using pretrained feature embeddings, and applies an **Unknown Rejection** mechanism to reject non-matching faces.

---

## 🌟 Key Features

- 👤 **Face Detection Wrapper**: Detects face locations using `face_recognition` (dlib HOG/CNN) with a native OpenCV Haar/DNN fallback.
- 🧬 **128-d Feature Embeddings**: Uses dlib's pretrained 128-D face embedding model when `face_recognition` is installed; otherwise uses the documented OpenCV spatial fallback descriptor.
- 📐 **Similarity Matching & Metrics**: Supports **Cosine Similarity** (default) and **Euclidean Distance (L2)** with normalized confidence percentage scoring ($0\%$ to $100\%$).
- 🚫 **Unknown Rejection Threshold**: Automatically classifies faces with similarity scores below a configurable threshold as `"Unknown"` instead of forcing false matches.
- 📊 **Evaluation & Threshold Tuning Sweep**: Evaluates accuracy, False Acceptance Rate (FAR), False Rejection Rate (FRR), Precision, Recall, and F1-score; plots ROC/Threshold trade-off curves and generates markdown reports.
- 🌐 **Modern Glassmorphism Web App**: Real-time image dropzone, live webcam camera snapshotting, canvas bounding box annotations, identity management gallery, and interactive evaluation dashboards.
- 💻 **CLI & API Interfaces**: Dedicated CLI tools (`enroll.py`, `identify.py`, `evaluate.py`) and REST API endpoints (`app.py`).

---

## 📁 Project Structure

```
face-recognition-system/
├── app.py                      # FastAPI REST API Backend server
├── config.py                   # System configuration & thresholds
├── create_sample_data.py       # Sample face dataset generator
├── enroll.py                   # Root entrypoint CLI for face enrollment
├── identify.py                 # Root entrypoint CLI for face identification
├── evaluate.py                 # Root entrypoint CLI for threshold evaluation sweep
├── database.py                 # Root entrypoint for face database management
├── detector.py                 # Root entrypoint for face detection
├── embedder.py                 # Root entrypoint for feature embedding extraction
├── matcher.py                  # Root entrypoint for similarity matching logic
├── face_utils.py               # Root entrypoint for image & face helper functions
├── requirements.txt            # Package dependencies
├── README.md                   # Documentation
├── data/
│   ├── enrolled_images/        # <person_name>/*.jpg enrollment photo subfolders
│   └── test_images/            # Query test images
├── database/
│   └── face_database.pkl       # Persistent pickle embedding database
├── results/
│   ├── evaluation_report.md    # Generated threshold sweep evaluation report
│   ├── threshold_curve.png     # FAR / FRR / Accuracy vs Threshold plot
│   └── annotated_*.jpg         # Output query images with bounding boxes & labels
├── static/
│   ├── index.html              # Modern Web Application HTML UI
│   ├── styles.css              # Glassmorphism dark-mode CSS theme
│   └── app.js                  # Frontend JavaScript for Web App & Webcam
├── src/                        # Core Python package implementations
│   ├── database.py
│   ├── detector.py
│   ├── embedder.py
│   ├── enroll.py
│   ├── evaluate.py
│   ├── face_utils.py
│   ├── identify.py
│   └── matcher.py
└── tests/                      # Automated pytest unit test suite
    ├── test_api.py
    ├── test_database.py
    ├── test_detector.py
    ├── test_embedder.py
    ├── test_matcher.py
    └── test_pipeline.py
```

---

## 🤖 Models & Approach

| Stage | Model / Technique | Vector Dimension / Specs | Rationale |
|---|---|---|---|
| **Face Detection** | dlib HOG/CNN when `face_recognition` is installed; otherwise OpenCV Haar cascade with eye-landmark validation | Bounding box `(top, right, bottom, left)` | CPU-friendly fallback for frontal and near-frontal views; invalid/no-face frames return no detections. |
| **Face Embedding** | dlib's pretrained 128-D model when available; otherwise normalized OpenCV spatial intensity/gradient descriptor | **128-dimensional** L2-normalized vector | The dlib path is a learned face embedding. The fallback is deterministic and suitable for local demos, but is not equivalent to a pretrained neural face model. |
| **Similarity Metric** | Cosine Similarity $S_{\text{cos}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | Range $[0.0, 1.0]$ | Invariant to vector scale, correlates directly with confidence percentage ($S_{\text{cos}} \times 100\%$). |
| **Unknown Rejection** | Threshold rule: $\text{Match}$ if $S_{\text{cos}} \ge \text{MATCH\_THRESHOLD}$ else `"Unknown"` | Configurable default: $0.50$ (Cosine) / $0.55$ (Euclidean) | Prevents forcing unknown faces into nearest incorrect identity. |

---

## ⚙️ Matching Method & Threshold Sweep

The decision rule for query embedding $\mathbf{q}$ against enrolled embeddings $\{\mathbf{e}_i\}_{i=1}^N$ with labels $\{y_i\}_{i=1}^N$:

$$i^* = \arg\max_{i=1..N} S_{\text{cos}}(\mathbf{q}, \mathbf{e}_i)$$

$$\text{Identity}(\mathbf{q}) = \begin{cases} y_{i^*} & \text{if } S_{\text{cos}}(\mathbf{q}, \mathbf{e}_{i^*}) \ge \tau \\ \text{"Unknown"} & \text{otherwise} \end{cases}$$

### Threshold Sweeping & Optimization
Threshold $\tau$ is tuned by running `evaluate.py`, which generates genuine (same identity) and impostor (different identity) pairs and sweeps thresholds from $\tau = 0.10$ to $0.95$. The recommended threshold maximizes measured accuracy; FAR/FRR balance and F1 are tie-breakers. Evaluation now refuses to report metrics when fewer than four valid pairs are available.

---

## 📊 Evaluation Results

Running `python evaluate.py` computes the metrics on the currently available validation pairs. The values are dataset-dependent and must not be treated as fixed benchmark results. The generated report is written to `results/evaluation_report.md`; the API returns HTTP 422 instead of a misleading zero-valued report when no valid face pairs are available.

| Threshold | Accuracy | FAR (False Accept) | FRR (False Reject) | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| See `results/evaluation_report.md` | Dataset-dependent | Dataset-dependent | Dataset-dependent | Dataset-dependent | Dataset-dependent | Dataset-dependent |

### Trade-off Curve
The evaluation script generates `results/threshold_curve.png`, visualizing Accuracy, FAR, and FRR across thresholds:

![Threshold Curve](results/threshold_curve.png)

The report is meaningful only when the dataset contains readable images with detected faces. Add multiple images per identity and at least two identities before evaluating.

---

## 🔬 Case Studies & Example Analysis

1. **True Positive Example**:
   - *Scenario*: Alice query image vs enrolled Alice reference photo.
   - *Metric*: The actual cosine similarity is recorded in the generated evaluation output.
   - *Result*: Correct only when the measured score reaches the configured threshold.
2. **False Positive Example**:
   - *Scenario*: Impostor query (Look-alike face or extreme shadow contrast).
   - *Metric*: An impostor score at or above the configured threshold is counted as FAR.
   - *Result*: Increase the threshold only after validating the FAR/FRR trade-off on representative data.
3. **False Negative Example**:
   - *Scenario*: Enrolled person viewed at extreme 60-degree profile angle or partially occluded by heavy dark sunglasses.
   - *Metric*: A genuine score below the configured threshold is counted as FRR.
   - *Result*: Multi-shot enrollment with varied pose and lighting can reduce this failure.

---

## ⚠️ Failure Cases & Limitations

- **Lighting & Contrast**: Extreme backlighting or underexposed low-light conditions impair feature extraction.
- **Extreme Pose & Angle**: Large yaw/pitch face rotations ($>45^\circ$) increase feature distance for the same person.
- **Occlusions**: Masks, heavy sunglasses, or hands covering facial landmarks reduce similarity.
- **Resolution & Motion Blur**: Low-resolution camera frames ($<64\times 64$ pixels) produce noisier embeddings.
- **Look-Alikes & Twins**: Identical twins or close relatives share high facial similarity.
- **Aging & Facial Alterations**: Dramatic age differences, facial hair changes, or heavy makeup between enrollment and query.

---

## 🚀 Future Improvements

- **Multi-Shot Embedding Averaging**: Compute mean normalized embedding per identity to increase prototype robustness.
- **Liveness & Anti-Spoofing**: Add real-time eye-blink detection and depth analysis to prevent photo/video spoofing attacks.
- **FAISS Vector Indexing**: Integrate Meta's FAISS library for sub-millisecond vector retrieval on datasets with $>100,000$ identities.
- **Face Alignment Preprocessing**: Apply 5-point landmark face alignment before feature extraction.

---

## 💻 Setup & Usage Instructions

### 1. Prerequisites & Installation

```bash
# Clone repository
cd face-recognition-system

# Create virtual environment
python -m venv venv
# Activate on Windows: venv\Scripts\activate
# Activate on macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python create_sample_data.py
```

### 3. Enroll Identities (CLI)

```bash
# Enroll all subfolders in data/enrolled_images/
python enroll.py

# Enroll single person
python enroll.py --name "Alice" --image path/to/photo.jpg

# List enrolled identities
python enroll.py --list

# Remove an identity
python enroll.py --remove "Alice"
```

### 4. Identify Query Faces (CLI)

```bash
# Identify faces and save annotated image
python identify.py data/test_images/alice_query.jpg --save --threshold 0.50 --metric cosine
```

### 5. Run Threshold Evaluation (CLI)

```bash
python evaluate.py
```

### 6. Launch Web Application & REST Server

```bash
python app.py
```
Open your browser at **`http://localhost:8000`** to access the interactive web interface!

### 7. Run Unit Test Suite

```bash
python -m pytest -v
```

---

## 🛡️ Privacy, Ethics & Responsible AI

- **Consent & Authorization**: Face identification systems must only be deployed with explicit consent from individuals being enrolled.
- **Biometric Data Protection**: Enrolled embeddings are stored as mathematical vector hashes (`.pkl`) rather than raw biometric imagery. Keep database files secured with appropriate access controls.
- **Bias Mitigation**: Ensure enrollment data represents diverse skin tones, ages, and facial characteristics to minimize algorithmic bias.
- **Non-Surveillance Use**: This software is designed strictly for access management, verification, and research purposes, not covert public surveillance.

---

## 📄 License

MIT License — see `LICENSE` for details.
