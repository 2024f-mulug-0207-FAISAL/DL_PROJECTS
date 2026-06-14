# Natural Language Product Sentiment Engine (Text-GRU RNN)

## 📌 Project Overview
Unstructured customer reviews contain massive amounts of textual feedback that cannot be parsed by standard linear matrix operations. This Natural Language Processing (NLP) framework establishes a deep sequence-processing pipeline using an optimized **Gated Recurrent Unit (GRU)** model to automatically track token sequences and identify positive or negative semantic orientation.

## 🧠 Technical Workflow & NLP Sequence Mapping
The structural text-processing pipeline maps text properties through distinct stages:

1. **Tokenization Vocabulary Dictionary:** Constructs an automated operational index map ($\text{vocab\_size}$) from all available raw text strings, assigning every unique word code string to a dedicated deterministic token index identifier.
2. **Sequential Padding:** Constrains variable-length customer review inputs to a strict text length sequence of 5 tokens. Shorter feedbacks are padded with neutral zero-tokens (`0`), while longer string reviews are cleanly truncated.
3. **Embedding Vectorization Layer (`nn.Embedding`):** Maps discrete text token indices into low-dimensional dense continuous vectors of size 16. Words with similar contextual semantic meanings organically cluster close together within this mathematical space.
4. **Gated Recurrent Unit Sequence Block (`nn.GRU`):** Processes the sequence vector series iteratively. GRUs condense memory state management down to two internal operational gates (Reset Gate and Update Gate), providing standard LSTM-quality memory processing with faster operational speeds.
5. **Inference Mapping:** Squeezes final recurrent hidden states directly into a linear binary layer wrapped in a Sigmoid activator to output reliable operational polarity probabilities.