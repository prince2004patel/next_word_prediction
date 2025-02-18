# Next Word Prediction Using LSTM + Attention

## Overview
This project implements a **Next Word Prediction** model using **LSTM (Long Short-Term Memory) with an Attention mechanism**. The model is trained on **Shakespeare's Hamlet dataset** and achieves **70% accuracy** in predicting the next word in a sequence.

## Project Workflow

### 1. Data Collection
- Collected **Shakespeare's Hamlet** dataset as raw text.

### 2. Data Preprocessing
- **Tokenization:** Used `Tokenizer` from Keras to convert text into sequences.
- **Lowercasing:** Converted all text to lowercase for consistency.
- **Vocabulary Size Check:** Identified a total of **4,818 unique words**.
- **Input Sequence Creation:**
  - Generated input sequences by taking a sliding window of words.
  - Example: "to be or not to" → "be or not to predict_next_word".
- **Finding Maximum Sequence Length:** Computed the longest input sequence length.
- **Padding Sequences:**
  - Used `pad_sequences()` to apply **pre-padding** (padding at the beginning) to standardize input size.

### 3. Feature Engineering
- **Divided dataset into X (input sequences) and y (target words).**
- **Performed train-test split** to prepare the dataset for model training.

### 4. Model Development
- Built an **LSTM-based neural network with an Attention mechanism**.
- Model architecture:
  - **Embedding Layer** to convert words into vector representations.
  - **LSTM Layer** to capture sequential dependencies.
  - **Attention Layer** to focus on important words in the sequence.
  - **Dense Output Layer** with a softmax activation for predicting the next word.
- **Compiled the model using categorical cross-entropy loss** and Adam optimizer.

### 5. Model Training
- Trained the model on the dataset.
- Achieved **70% accuracy** on test data.

### 6. Prediction Function
- Created a **helper function** for predicting the next word:
  - Takes input text.
  - Tokenizes and pads it to match the model's input shape.
  - Predicts the most likely next word.
- Successfully tested predictions on different inputs.

### 7. Streamlit Web Application
- Built a **Streamlit-based UI** to interact with the model.
- Users can enter a sentence, and the model predicts the next word.

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the **localhost URL** to interact with the app.

## Technologies Used
- **Python**
- **TensorFlow & Keras** (Deep Learning)
- **Streamlit** (Web Interface)
- **Numpy ,Pandas & Pickle** (Data Handling)

## Future Improvements
- Train on a **larger dataset** for improved generalization.
- Experiment with **transformer-based models (e.g., GPT, BERT)** for better predictions.
- Optimize the **attention mechanism** to enhance word predictions.
