# Structural Steel Fault Detection: Traditional Machine Learning vs Deep Learning

**Author:** Edwin Bayingana  
**Course:** Intro to Machine Learning - Summative

### Project Overview
The structural integrity of modern infrastructure relies fundamentally on the quality of the materials deployed during construction. I developed this project to bridge the gap between material science and artificial intelligence. 

This repository contains the complete codebase, evaluation dashboards, and supplementary materials for my Summative Machine Learning Project. The primary objective is to automate the detection of structural steel plate defects by systematically comparing traditional ensemble algorithms against deep neural networks.

### Dataset Source
The data utilized in this project is the **Steel Plates Faults Dataset**, publicly provided by the UCI Machine Learning Repository. 
* **Link:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
* **Description:** The dataset contains 1941 instances of steel plates, classified into 7 distinct fault categories (Pastry, Z Scratch, K Scatch, Stains, Dirtiness, Bumps, and Other Faults) using 27 numeric feature columns representing geometric and radiometric properties.

### Repository Structure
* `Edwin_Steel_Fault_Analysis.ipynb` : The main Jupyter Notebook containing the entire end to end machine learning pipeline.
* `README.md` : Project documentation and reproduction instructions.

### Methodology and Execution
This project features a rigorous comparative pipeline consisting of sixteen systematically varied experiments. 

**Key Technical Highlights:**
* **Data Leakage Prevention:** A strict stratified split divides the data into Training (70 percent), Validation (15 percent), and Testing (15 percent) sets. Standard scalers were fitted exclusively on the training data.
* **Class Imbalance Handling:** The Synthetic Minority Oversampling Technique (SMOTE) was applied exclusively to the training set to generate synthetic instances of rare structural defects, ensuring the model could identify rare but critical faults.
* **Algorithmic Comparison:** Eight Random Forest ensemble configurations were benchmarked against eight distinct Deep Learning architectures built with the TensorFlow Keras Sequential API.
* **Evaluation Dashboard:** Every experiment generates a unified dashboard featuring a Classification Report, a Confusion Matrix, and Multi Class ROC AUC curves.

### Instructions for Reproducibility 
This codebase is designed for perfect reproducibility. The notebook runs sequentially from top to bottom without errors. 
1. Open `Edwin_Steel_Fault_Analysis.ipynb` in Google Colab or a local Jupyter environment.
2. The first cell contains a `set_seeds` function that locks the Python, NumPy, and TensorFlow random states to a constant integer (42). 
3. Ensure all required standard data science packages are installed (Pandas, NumPy, Scikit Learn, Imbalanced Learn, TensorFlow, Matplotlib, Seaborn).
4. Run all cells sequentially. The final cell will automatically evaluate the winning model against the locked test set and print the final evaluation metrics.

### Key Findings
The experimental evaluation concluded that for highly structured tabular datasets with limited sample sizes, traditional ensemble methods significantly outperform heavily regularized deep neural networks. The winning model, a Random Forest classifier utilizing the entire dataset without bootstrapping, achieved a 74 percent global accuracy on the completely unseen test set. It demonstrated near perfect mathematical isolation of extreme structural defects like bumps and severe scratches, proving its viability for industrial quality control applications.
