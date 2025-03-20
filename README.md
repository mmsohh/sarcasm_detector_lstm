# sarcasm_detector_lstm
Developed an LSTM-based sarcasm detection model that leverages deep learning and NLP techniques to classify news headlines, highlighting AI’s ability to interpret linguistic nuances.  Achieved a 87.84% Validation Accuracy, 0.3389 Validation Loss, and 95.55% accuracy when tested on entire dataset. 

# 🧠 Sarcasm Detection Using LSTM

## 🚀 Motivation & Project Overview  
AI, specifically text generator AIs, can summarize and reword news articles, but how well can AI **understand nuanced sentiment**?  

Sarcasm is unique because it often **contradicts the literal meaning** of the words used. Unlike traditional sentiment analysis models, which classify emotions as **positive, negative, or neutral**, sarcasm detection requires **contextual awareness** and a deeper understanding of language to recognize irony.  

This project focuses on developing an **NLP sarcasm detection model** using the **"News Headlines Dataset For Sarcasm Detection"** from Kaggle, created by **Rishabh Misra**, containing **26,000+ labeled news headlines**.  

The dataset is sourced from:  
- 📰 **The Onion** (*satirical news, sarcastic headlines*)  
- 📰 **HuffPost** (*non-satirical, factual headlines*)  

For this project, I implemented a **Long Short-Term Memory (LSTM) network**, a specialized type of **Recurrent Neural Network (RNN)** designed to process sequential text data. Unlike traditional neural networks, LSTMs **retain memory** from previous inputs, making them effective for understanding **sentence structure and context**—both crucial for sarcasm detection.

## 📚 Dataset Source  

This project utilizes the **"News Headlines Dataset For Sarcasm Detection"**, originally published by **Rishabh Misra**.  

### 🔗 **Citations:**  
- Misra, Rishabh, and Prahal Arora. **"Sarcasm Detection using News Headlines Dataset."** *AI Open* (2023).  
- Misra, Rishabh, and Jigyasa Grover. **"Sculpting Data for ML: The First Act of Machine Learning."** ISBN 9798585463570 (2021).  

---

## 📌 Project Workflow  

### 🔍 **Data Analysis & Preprocessing**  
✔️ Loaded and analyzed the dataset, examining the **distribution of sarcastic vs. non-sarcastic headlines**.  
✔️ Extracted the **most frequently used words** in each category to identify patterns.  
✔️ Created a **word cloud visualization** to highlight the most common words in sarcastic vs. non-sarcastic headlines.  

---

### 🏗 **Building the LSTM Model**  
✔️ **LSTM Layers** to retain sequential context and capture sentence structure.  
✔️ **Dropout** to prevent overfitting by randomly deactivating neurons during training.  
✔️ **Dense Layers** to refine extracted features for classification.  
✔️ **ReLU Activation** in dense layers for efficient learning.  
✔️ **Sigmoid Activation** in the output layer to produce a **probability score** for sarcasm detection.  

---

### ⚙ **Hyperparameter Tuning**  
This project wasn’t just about optimizing accuracy, but also about understanding **how different hyperparameters impact** the LSTM model by testing:  

✔️ **Train/Test Split Ratios** → Finding the best balance of training vs. validation data.  
✔️ **Dropout Rates** → Preventing overfitting while maintaining model performance.  
✔️ **L1 & L2 Regularization** → Controlling complexity and reducing reliance on specific features.  
✔️ **Number of LSTM Layers** → Testing if increasing depth improves performance.  
✔️ **Dense Layers & Neurons** → Exploring the trade-off between model complexity and efficiency.  

After tuning, I combined my findings to construct an **optimized model**.  

---

## 🏆 Best Model Overview  
The **best-performing LSTM sarcasm detection model** achieved an accuracy of **87.84%** on the test set. This configuration optimized key hyperparameters to enhance generalization and performance.  

### **📌 Model Configuration:**  
- **Dropout:** 0.6  
- **L1 Regularization:** 0.01  
- **Train/Test Split:** 5/95  
- **Architecture:** Maintained the original LSTM and Dense layers  
- **Saved Model:** `best_sarcasm_LSTM_model3.keras`  

### **📊 Training & Validation Metrics:**  
| Metric            | Value   |
|------------------|--------|
| **Training Accuracy** | 92.46% |
| **Training Loss** | 0.2845 |
| **Validation Accuracy** | 87.84% |
| **Validation Loss** | 0.3389 |

When tested on the **entire dataset**, the model achieved an accuracy of **95.55%**, which was expected since it was trained on this data.

---

## 🔬 Real-World Headline Evaluation  
To further validate the model, I tested it on **previously unseen headlines** to evaluate real-world performance. These headlines include both **sarcastic** and **non-sarcastic** examples from current events and manually written statements.

### **📰 Sample Headlines & Model Predictions:**  
✅ *"A new hurricane is approaching East Atlantic."* → **Non-Sarcastic (Correct)**  
✅ *"Breaking: Local Man Shocked to Discover Monday Comes Every Week."* → **Sarcastic (Correct)**  
✅ *"Experts Warn That Doing Nothing Will Definitely Fix the Economy."* → **Sarcastic (Correct)**  
✅ *"Donald Trump executes tariffs for the U.S."* → **Non-Sarcastic (Correct)**  
✅ *"Study Finds 100% of People Eventually Die."* → **Sarcastic (Correct)**  
✅ *"Brilliant Political Plan Solves Everything, Announces Nobody."* → **Sarcastic (Correct)**  
❌ *"New York Times columnist admits scientists ‘badly misled’ public on COVID-19: ‘Five years too late’."* → **Misclassified (Should be Non-Sarcastic)**  
✅ *"Greenpeace must pay over \$660M in case over Dakota Access protest activities, jury finds."* → **Non-Sarcastic (Correct)**  
✅ *"Trump administration says it's cutting \$175 million in funding to the University of Pennsylvania."* → **Non-Sarcastic (Correct)**  
✅ *"Forgetful Man Playing Fast And Loose With Free Trials."* → **Sarcastic (Correct)**  

### **🤔 Observations:**  
The model **correctly classified 9 out of 10 headlines**. The only misclassification occurred with:  
- **Misclassified Headline:** *"New York Times columnist admits scientists ‘badly misled’ public on COVID-19: ‘Five years too late’."*  
- This misclassification is interesting because **while the statement is intended as straightforward news, it has a tone that could be interpreted as sarcastic**, possibly explaining the model's error.

---

## 🛠 Skills Applied  
✔️ **Deep Learning & Neural Networks**: Gained expertise in **LSTMs** for sarcasm detection, including **sequential data processing** and **hyperparameter tuning**.  
✔️ **Natural Language Processing (NLP)**: Applied **text tokenization, stopword analysis, and word embeddings** to process textual data.  
✔️ **Model Architecture Design**: Implemented **stacked LSTM layers, Dense layers, and Dropout** for sarcasm detection.  
✔️ **Overfitting Prevention & Regularization**: Experimented with **Dropout rates, L1 & L2 regularization, learning rates, and train/test split sizes** to improve generalization.  

---

## 🏁 Final Thoughts  
This project **showcases the intersection of AI and linguistics**, demonstrating how **LSTM-based machine learning models can classify sarcasm** and enhance sentiment analysis in real-world text processing.  

---

## 📂 Repository Contents  
- 📜 **`sarcasm_detection_lstm.ipynb`** → Jupyter Notebook with model training, evaluation, and visualizations.  
- 📊 **`word_clouds_including_stop_words.png`** → Word cloud visualizations of sarcastic vs. non-sarcastic words (including stop words).
- 📊 **`word_clouds_excluding_stop_words.png`** → Word cloud visualizations of sarcastic vs. non-sarcastic words (excluding stop words).  
- 📄 **`best_sarcasm_LSTM_model3.keras`** → Saved model file for reuse.  
---

## 📢 Contact  
For questions or improvements, feel free to open an issue or contribute!  
📚 Personal Website: www.manreetsohi.com  

