# MLOps Intro Workshop

Welcome! This repository contains the material and exercises for our **MLOps workshop**.
You’ll learn the foundations of **reproducibility, version control, and model packaging**, key skills for any applied ML project.

---

## Workshop Overview

### Session 2 — Packaging and Documentation for Production

At the end of this lecture, you will be able to:

- Export trained models using joblib, pickle, and torch.save.
- Install and use Docker to build and run local images.
- Document a model’s inputs and outputs for future API integration.
- Apply best practices for reproducibility and scaling toward MLOps.
- Complete a mini-challenge where you package a trained model, document it, and publish it on GitHub.

---

## 📁 Repository Structure

```
mlops-workshop/
├── mnist_classification/          # Example: digit classification (PyTorch)
│   ├── data/
│   ├── src/
│   │   ├── train.py
│   │   └── predict.py
│   ├── models/
│   ├── environment.yml
│   ├── Dockerfile
│   └── README.md
│
├── heart_failure_prediction/      # Example: heart failure prediction (Random Forest)
│   ├── data/
│   ├── src/
│   │   ├── train.py
│   │   └── predict.py
│   ├── models/
│   ├── environment.yml
│   ├── Dockerfile
│   └── README.md
│
├── submissions/                   # Student submissions
│   ├── dog_vs_cat/                # Image classification challenge
│   ├── heart_failure_prediction/  # Heart failure model submissions
│   ├── mnist_classification/      # MNIST model submissions
│   └── text_classification/       # NLP challenge (spam, emotion, etc.)
│
└── README.md                      # You are here
```

---

## In-Class Exercises

Each student will:
- Fork this repository to their own GitHub account.
- Work on the two guided exercises: mnist_classification/ and heart_failure_prediction/.
- Submit their work by creating a Pull Request (PR) to this repository.

**Naming convention for branches:**  
firstname-lastname/project

Example:

maria-gomez/heart_failure_prediction
