# Touché 2025 Task 4 Participant Approaches

This note summarizes the main participant approaches for **Touché 2025, Task 4: Advertisement in Retrieval-Augmented Generation**.

## Task Structure

Task 4 introduced two subtasks:

1. **Subtask 1**: generate RAG responses with embedded advertisements
2. **Subtask 2**: detect whether a response contains an advertisement

Four teams participated. Most systems combined:

- LLM-based RAG pipelines for advertisement generation
- fine-tuned encoder classifiers for advertisement detection

## Participant Approaches

### JU-NLP

- Used preference-based fine-tuning with **ORPO** on **Mistral-7B** for stealth advertisement generation
- Used **Cross-Encoder (MPNet)** and **DeBERTa** models for advertisement detection
- Achieved near-perfect detection performance with **F1 around 1.0**

Key idea:

- preference-tuned stealth advertisement generation

### TeamCMU

- Built an **ad rewriter + ad classifier** pipeline
- Used **Best-of-N sampling** to minimize detectability
- Created synthetic hard positive and hard negative training examples

Key idea:

- classifier-guided adversarial ad insertion

### Pirate Passau

- Compared several detection approaches:
- **TF-IDF + Random Forest**
- **Sentence Transformers**
- **LLM few-shot prompting**
- **RAG classifier**
- Best result came from a fine-tuned **MiniLM** model with **F1 around 0.97**

Key idea:

- fine-tuned sentence transformers outperform LLM prompting

## Overall Findings

- Best detection systems were based on **DeBERTa** and **sentence transformer** models
- Strong generation systems used **Mistral** and **Qwen** RAG pipelines
- Classifier-guided generation produced the hardest-to-detect ads
- Classical embedding-based models were strong baselines

## Relevance For This Repository

These 2025 systems provide the immediate baseline context for the 2026 work in this repository:

- encoder-based classifiers remain a strong reference point for Subtask 1
- adversarial or stealth-oriented generation helps explain why neutral-rewrite and semantic-delta methods are interesting
- the 2025 findings motivate comparing plain classifier inputs against response-vs-neutral semantic features
