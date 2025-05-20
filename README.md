# Sequence to Sequence Transliteration Model with Attention

## Overview

This project implements a character-level sequence-to-sequence (seq2seq) model with attention for transliteration, using the Dakshina dataset released by Google. The goal is to learn how to model seq2seq problems using recurrent neural networks (RNN, GRU, LSTM), compare different cell types, incorporate attention to improve performance, and visualize the model's internal behavior.

---

## Dataset

The dataset contains transliteration pairs of the form:

```
<target_language_word>\t<source_language_word>\t<number>
```

For example:

```
ajanabee    अजनबी   123
```

Here, the first word is the target transliteration (in Latin script), the second is the source word (in Devanagari/Hindi script), and the third is an auxiliary number which can be ignored.

---

## Methodology

- **Data preprocessing:** Read TSV files, build vocabularies for source and target scripts, encode sequences as integer token sequences.
- **Model:** Encoder-decoder architecture with RNN-based cells (RNN/GRU/LSTM). Embeddings for source and target vocabularies.
- **Attention:** Added additive attention mechanism to improve decoder context awareness.
- **Training:** CrossEntropyLoss with padding ignored, teacher forcing during training, evaluation with accuracy metrics.
- **Sweeps:** Hyperparameter optimization using Weights & Biases sweeps (embedding size, hidden size, cell type, dropout, learning rate, batch size).

---

## Project Structure

```
Assignment_3/
│
├── main.py                 # Main training script with wandb integration
├── train.py                # Training and evaluation functions
├── model.py                # Seq2Seq model implementation with attention
├── dataset.py              # Dataset class and collate function for batching
├── vocab.py                # Vocabulary building and encoding utilities
├── data_utils.py           # TSV reading and helper functions
├── sweep.yaml              # WandB sweep configuration file
├── src_stoi.json           # Source language token-to-index map (saved after training)
├── tgt_itos.json           # Target language index-to-token map (saved after training)
├── wandb/                  # WandB logs and sweep files
│
└── dakshina_dataset_v1.0/  # Dataset folder (download and place dataset here)
```

---

## Requirements

- Python 3.7+
- PyTorch
- tqdm
- wandb

Install dependencies:

```bash
pip install torch tqdm wandb
```

---

## How to Run

1. **Download the Dakshina dataset**
  
  Download from [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina) and place the relevant Hindi lexicons in:
  
  ```
  dakshina_dataset_v1.0/hi/lexicons/
  ```
  
2. **Prepare your sweep configuration**
  
  Edit `sweep.yaml` to set desired hyperparameters and file paths.
  
3. **Start a WandB sweep**
  
  Initialize the sweep:
  
  ```bash
  wandb sweep sweep.yaml
  ```
  
  Then start an agent to run the sweep:
  
  ```bash
  wandb agent <sweep_id>
  ```
  
4. **Run training manually**
  
  You can also run a single training run manually:
  
  ```bash
  python main.py
  ```
  
5. **Check logs and models**
  
  Models are saved after training as `model.pth`.
  
  Vocabulary files `src_stoi.json` and `tgt_itos.json` are saved for inference use.
  
6. **Modify hyperparameters**
  
  Use the `sweep.yaml` to experiment with different embedding sizes, hidden units, RNN cell types, dropout rates, and learning rates.
  

---

## Notes

- Make sure the target is the first word and source the second word when loading dataset lines.
- Padding tokens are handled in loss and accuracy computation.
- Attention is integrated in the model to help the decoder focus on relevant encoder states.

---


## Methodology

This project focuses on building a sequence-to-sequence (Seq2Seq) transliteration model using Recurrent Neural Networks (RNNs) with an attention mechanism to improve performance. The main goals were:

1. **Modeling sequence-to-sequence learning problems** using vanilla RNN, GRU, and LSTM cells.
2. **Comparing different RNN cell types** to evaluate their impact on model accuracy and convergence.
3. **Incorporating attention networks** to overcome limitations of traditional Seq2Seq models, particularly in handling longer sequences by allowing the decoder to selectively focus on relevant parts of the input sequence.
4. **Visualizing model components and training dynamics** using Weights & Biases (wandb) for effective hyperparameter tuning via sweeps.

The dataset used is the Dakshina corpus, which contains parallel transliteration pairs with the format:

```
<target_language_word>  <source_language_word>  <some_number>
```

The model performs character-level transliteration from the source language to the target language.

---


## File Descriptions

* **data\_utils.py**
  Contains functions to load the dataset TSV files, parse them, and build vocabulary dictionaries (`stoi` and `itos`) for both source and target languages.

* **dataset.py**
  Implements a custom PyTorch `Dataset` class `TransliterationDataset` which takes transliteration pairs and converts them into sequences of token IDs. Also contains the `collate_fn` function to pad sequences for batching.

* **model.py**
  Defines the `Seq2Seq` model class which contains the encoder and decoder RNN cells. The decoder incorporates an attention mechanism that computes attention weights over the encoder outputs, enabling the decoder to focus on relevant input tokens dynamically. Supports RNN, GRU, and LSTM cells.

* **train.py**
  Contains the training logic for one epoch (`train_one_epoch`), including the forward pass, loss computation, backward pass, optimizer step, and accuracy calculation.

* **main.py**
  The main script that initializes wandb, loads data, creates datasets and dataloaders, builds the model with specified hyperparameters, and orchestrates the training and validation loops. It logs metrics to wandb, saves the best model checkpoint, and exports vocabulary mappings.

* **vocab.py**
  Utility functions for encoding sequences of characters into integer indices using the vocab dictionaries and decoding back to characters.

* **wandb\_sweep.yaml**
  Configuration file for wandb hyperparameter sweeps, specifying the project name, program to run, search method, metric to optimize, and hyperparameter ranges.

* **README.md**
  The project documentation explaining methodology, project structure, and instructions on how to run the training and sweeps.

---

#### Github Repository: [GitHub Repository Link](https://github.com/Ishaan-Maheshwari/Assignment_3)
#### WandB Project: [WandB Project Link](https://wandb.ai/ishaan_maheshwari-indian-institute-of-technology-madras/seq2seq_transliteration/reports/Ishaan-Maheshwar-s-Assignment-3--VmlldzoxMjg1NDA3MA?accessToken=u9hsrq1whvp9sijyd4n7zfab1c66yq3r5f0us8ibnn8ya51z8d16k73wiq4ghpiy)
