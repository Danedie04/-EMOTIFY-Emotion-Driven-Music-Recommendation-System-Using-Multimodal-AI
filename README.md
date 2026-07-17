# 🎵 Music Emotion Analytics Pipeline
### Contextual Audio Clustering & Multi-Model Inference for Digital Streaming Platforms
**Project Lead:** Dineshkumar

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google-gemini&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow?style=for-the-badge)

---

## 🎯 Business & Technical Overview

The **Music Emotion Analytics Pipeline** is an enterprise-grade, object-oriented data science framework engineered to bridge the gap between raw track metadata and real-time business intelligence.

* **The Challenge:** Digital streaming services face a critical hurdle with contextual ad placement. Traditional ad servers deliver content blindly, causing emotional friction by dropping loud or upbeat advertisements into a listener's calm, ambient, or melancholic playlist.
* **The Solution:** This pipeline automates the ingestion, sanitization, and mathematical modeling of audio attributes using the Geneva Emotional Music Scales (**GEMS**) framework. By transforming raw tracking points into deterministic emotional signals, streaming applications can dynamically serve contextually aligned B2B advertisements—minimizing user bounce rates, optimizing marketing ROI, and maximizing session retention.

---

## 🏗️ Core System Architecture

The codebase is built within a highly resilient, modular Python class architecture designed to execute end-to-end under a single runtime instantiation:


graph TD
    A[Data Ingestion: KaggleHub API] --> B[Data Engineering: Schema Sanitization & Imputation]
    B --> C[Feature Engineering: GEMS Scale & Emotional Density Mapping]
    C --> D[Unsupervised Layer: KMeans & Ward Hierarchical Tree]
    C --> E[Supervised Learning Matrix: Comparative Sweep]
    D --> F[Cognitive Layer: HF Text Parsing & Gemini 2.5-flash Strategy Execution]
    E --> F
    
1. **Automated Data Ingestion:** Connects securely to the `kagglehub` API to pull the authoritative Emotify classification dataset (8,407 rows, 17 columns) with localized caching.
2. **Feature Engineering & Sanitization:** Standardizes matrix column headers, handles missing attributes via automated median imputation, and synthesizes a continuous `emotional_density` vector.
3. **Unsupervised Clustering Engine:** Applies Spatial KMeans segmentation ($k=4$) alongside a Ward-linkage hierarchical dendrogram mapping to discover structural grouping boundaries across latent musical profiles.
4. **Supervised Analytics Sweep:** Evaluates a multi-classifier framework against an isolated regression feature space to guarantee predictive scale.
5. **Cognitive GenAI Integration:** Leverages a deep learning transformer for textual latent context identification, passing structured payloads into the modern `google-genai` SDK for automated executive brief generation.

---

## 📊 Model Performance & Production Validation

### Leakage Isolation & Validation Integrity

Early architecture models exposed a systemic target leakage risk where look-ahead metrics skewed evaluation accuracy. This production build isolates the independent feature matrix from downstream target assignments, yielding verified, mathematically sound metrics.

### 📈 Continuous Optimization (Regression Matrix)

* **Target Objective:** Predictive Track Intensity (`emotional_density`)
* **Evaluation Baseline:** Linear Regression Model
* **Production Results:** `Mean Squared Error (MSE): 0.7065` | `R2 Score: 0.0530`

### 📉 Comparative Classification Profile (9-Class Emotion Mapping)

With 9 distinct subjective target categories derived from user interaction metadata, a random guess baseline yields an accuracy of **11.1%**.

| Predictive Model Architecture | Evaluation Accuracy | Operational Lift vs. Baseline |
| --- | --- | --- |
| **Random Forest Ensemble** | **27.88%** | **2.51x Outperformance (Champion Build)** |
| Decision Tree Classifier | 27.59% | 2.48x Outperformance |
| Logistic Regression (L2 Regularized) | 26.69% | 2.40x Outperformance |
| K-Nearest Neighbors ($k=5$) | 23.45% | 2.11x Outperformance |
| Naive Bayes (Gaussian Core) | 18.36% | 1.65x Outperformance |

---

## 🤖 Cognitive NLP & Generative AI Output

The pipeline deploys a multi-stage cognitive inference layer to generate analytical value straight from model outcomes:

* **Latent Space Classification:** Runs a Hugging Face zero-shot pipeline (`DistilBERT-base-uncased`) to map raw track summaries to high-level categorical emotions. Upbeat test samples map to a `joyful` anchor label with **82.5% confidence**.
* **Dynamic B2B Strategic Engine:** Interacts natively with the production-ready **Gemini 2.5-flash** engine. Rather than relying on rigid, pre-authored statements, the system securely accesses Google Colab runtime secrets to output an automated, customized business pitch:

> ### 📋 Executive Insight Briefing (Generated Live)
> 
> 
> *"Our advanced musical emotion classification models enable Spotify to precisely align ad content with a listener's current emotional state. By understanding the emotional context of the music, ads are delivered at psychologically opportune moments, significantly boosting receptiveness and recall. This targeted approach not only optimizes ad placement for maximum ROI but also enhances the overall user experience, driving higher engagement across the platform."*

---

## 📁 Repository Directory Structure

```text
├── music_pipeline.py            # Production-ready, object-oriented python pipeline script
├── genre_distribution.png       # Cached distribution visualization from data engineering phase
├── hierarchical_dendrogram.png  # Ward-linkage hierarchical validation plot from clustering phase
└── README.md                    # Core project documentation and systems briefing

```

---

## 🚀 Getting Started & Environment Configuration

### Dependencies

Install the verified software dependencies from your terminal:

```bash
pip install kagglehub numpy pandas matplotlib seaborn scikit-learn google-genai transformers scipy

```

### API Key Initialization

To enable the Generative AI engine, save your secure token inside your deployment environment variables:

**Standard Linux/macOS Environment:**

```bash
export GEMINI_API_KEY="AIzaSyYourActualProductionAPIKeyHere"

```

**Google Colab Notebook Deployment:**

1. Open the left sidebar and click the **Secrets (Key icon)** tab.
2. Add a new secret with the name `GEMINI_API_KEY`.
3. Paste your key in the value input and toggle **Notebook access** to ON.

### Execution

Trigger the complete processing pipeline end-to-end:

```bash
python music_pipeline.py

```

