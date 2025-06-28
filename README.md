# SMART-HIRING-AI-Based-Resume-Screening\_Model

## Fair and Transparent AI-Driven Resume Screening System

---

## Project Overview

This project presents a **fair, accurate, and interpretable AI-powered resume screening system** designed to reduce bias in automated hiring. It integrates advanced NLP methods such as **BERT embeddings**, along with **bias detection and mitigation techniques** and **Explainable AI (XAI)** approaches to promote ethical and transparent recruitment decisions.

---

## Core Features

* **Data Preprocessing**: Cleaned and standardized resumes, extracted textual features, and incorporated demographic attributes (e.g., Gender, Race, Age Group) to enable bias assessment.
* **Feature Extraction**: Utilized **BERT embeddings** to effectively capture semantic resume content.
* **Bias Detection Metrics**:

  * **Statistical Parity**: Assesses equal selection rates across demographic groups.
  * **Disparate Impact Ratio**: Measures fairness between protected and unprotected groups.
  * **Intersectional Analysis**: Detects bias involving combinations of attributes (e.g., Gender and Race).
* **Bias Mitigation Strategies**:

  * **Counterfactual Fairness**: Ensures model predictions remain stable under hypothetical changes in sensitive attributes.
  * **Adversarial Debiasing**: Trains embeddings to minimize encoded sensitive information.
  * **Reweighing**: Balances dataset representation by adjusting sample weights.
* **Explainable AI**:

  * **LIME**: Provides interpretable explanations by highlighting key words or phrases influencing individual predictions.

---

## Model Files

Due to their size, saved model files are not included in this repository. You can download them here:
[Google Drive - Model Files](https://drive.google.com/drive/folders/1JGFqvBZBxatlnBK3EIKgoEtY2Nt8_mnk?usp=drive_link)

---

## Prerequisites

* Python 3.8 or higher
* Required libraries: PyTorch, Transformers, scikit-learn, pandas, numpy, seaborn, matplotlib

