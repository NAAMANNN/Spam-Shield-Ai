Spam Shield AI 🛡️

A robust, real-time machine learning application built with Streamlit that detects spam emails using Support Vector Machine (SVM) algorithms. This project demonstrates an end-to-end ML pipeline, from data ingestion (or simulation) to a user-friendly frontend.

📖 Project Summary

Spam Shield AI is designed to bridge the gap between raw machine learning scripts and interactive web applications. It serves as a practical demonstration of Natural Language Processing (NLP) concepts, specifically how Support Vector Machines (SVM) can classify text based on word frequency vectors.

Unlike static notebooks, this application allows users to:

Interact with the Model: Type in custom emails and get instant predictions.

Visualize Performance: See how changing the training data split impacts accuracy in real-time.

Handle Data Gracefully: The app includes a "Simulation Mode" that generates synthetic data if the primary CSV is missing, ensuring the app never crashes during a demo.

🚀 Features

Dual Data Mode: Automatically switches between a real dataset (emails.csv) and a built-in dummy data generator for seamless testing without external files.

Interactive Dashboard: Real-time text analysis with instant visual feedback (Safe vs. Spam).

Model Tuning: Sidebar controls to adjust hyperparameters like the training/testing data split.

Visual Analytics: Charts and metrics displaying model accuracy and dataset distribution.

🛠️ Technologies Used

Frontend: Streamlit

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn (SVM Classifier)

Visualization: Streamlit native charts

📂 Project Structure

spam-shield-ai/
├── app.py              # Main application logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── emails.csv          # (Optional) Dataset file


🏁 Getting Started

Prerequisites

Ensure you have Python 3.8+ installed.

Installation

Clone the repository:

git clone [https://github.com/yourusername/spam-shield-ai.git](https://github.com/yourusername/spam-shield-ai.git)
cd spam-shield-ai


Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py
