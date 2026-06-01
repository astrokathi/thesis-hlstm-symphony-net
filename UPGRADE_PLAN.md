# Accuracy Upgrade Plan

This document outlines the strategy for improving the accuracy and musicality of the HLSTM music generation model. This plan will be refined upon analysis of the "Advanced Scope" and "Future Scope" sections of the thesis.

## 1. Model Architecture Enhancements
- **Attention Mechanisms**: Integrate self-attention layers into the HLSTM to allow the model to capture long-range dependencies more effectively than standard LSTMs.
- **Transformer Integration**: Evaluate the transition to a Transformer-based architecture (e.g., Music Transformer) which is state-of-the-art for symbolic music generation.
- **Embedding Refinement**: Explore pre-trained music embeddings or learned embeddings that better represent musical relationships (e.g., circle of fifths).

## 2. Training Process Optimizations
- **Learning Rate Scheduling**: Implement a scheduler (e.g., Cosine Annealing with Warmup) to improve convergence and avoid local minima.
- **Gradient Clipping**: Apply gradient clipping to stabilize training and prevent exploding gradients, a common issue in deep LSTMs.
- **Regularization**: Experiment with varied Dropout rates and L2 regularization (Weight Decay) to prevent overfitting.
- **Batch Size & Epochs**: Systematically tune batch size and number of epochs based on validation loss plateaus.

## 3. Data Strategy & Augmentation
- **Dataset Expansion**: Increase the volume and variety of the training data from SymphonyNet.
- **Data Augmentation**: Implement musical augmentations such as:
    - **Transposition**: Shift songs to different keys to teach the model key-invariant patterns.
    - **Time Stretching**: Vary the tempo to improve duration predictions.
- **Tokenization Improvements**: Refine the event-based representation to better capture polyphony and complex rhythms.

## 4. Loss & Evaluation Metrics
- **Weighted Loss**: Use weighted Cross-Entropy loss to address class imbalance (e.g., some pitches or durations are more frequent than others).
- **Musicality Metrics**: Move beyond perplexity and incorporate musical metrics such as pitch histogram correlation and rhythmic consistency.

## 5. Hyperparameter Optimization
- **Automated Tuning**: Use tools like Optuna or Ray Tune to optimize:
    - `EMBED_DIM`
    - `HIDDEN_DIM`
    - `DROPOUT`
    - `LR`
