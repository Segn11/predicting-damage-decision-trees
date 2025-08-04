# 🏚️ Predicting Damage with Decision Trees

This project builds a Decision Tree Classifier to predict **severe building damage** based on structural attributes. Using real-world data from post-earthquake Nepal, the model classifies buildings into binary categories: **severely damaged** or **not**.

---

## 📌 Overview

- **Objective**: Predict whether a building suffered severe damage (Grade > 3) during an earthquake.
- **Model**: Decision Tree Classifier
- **Data Source**: Relational database (`nepal.sqlite`) with multiple joined tables
- **Scope**: Focused on buildings in District 4

---

## 🧠 Problem Statement

After a natural disaster like an earthquake, prioritizing damaged structures for inspection or aid is critical. This project aims to automate part of that process by predicting severe structural damage using features like:

- Building material
- Number of floors
- Foundation type
- Roof type, etc.

---

## 📊 Performance

| Metric              | Score |
|---------------------|-------|
| **Baseline Accuracy**  | 0.64  |
| **Training Accuracy**  | 0.72  |
| **Validation Accuracy**| 0.72  |
| **Test Accuracy**      | 0.72  |

> The model significantly improves over the baseline, while avoiding overfitting.



