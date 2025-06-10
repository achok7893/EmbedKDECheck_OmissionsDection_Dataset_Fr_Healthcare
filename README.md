# EmbedKDECheck_OmissionsDection_Dataset_Fr_Healthcare

# 🏥 French Synthetic Medical Reports and Summaries with Omission Labels (Fr-Healthcare)

📄 **Main Paper**: [Détection des omissions dans les résumés médicaux générés parles grands modèles de langue](https://arxiv.org/abs/your-paper-link)  
*Please cite this work if you use the dataset.*

---

## 🚩 Introduction: Detecting Omissions in LLM-Generated Medical Summaries

Large Language Models (LLMs) are increasingly used to summarize medical texts, but they may omit critical information, potentially compromising clinical decision-making. Unlike hallucinations, omissions involve the absence of essential facts.

This work introduces a validated French dataset for detecting such omissions and proposes **EmbedKDECheck**, a lightweight, reference-free detection method.

Unlike LLM-based methods, EmbedKDECheck leverages lexical embeddings from a lightweight NLP model combining FastText and Word2Vec using a specific algorithm coupled with an unsupervised anomaly scoring model. This approach efficiently identifies omissions with low computational cost.

EmbedKDECheck was evaluated against state-of-the-art frameworks (SelfCheckGPT, ChainPoll, G-Eval, GPTScore) and demonstrated strong performance. Our method enhances the reliability assessment of LLM outputs and contributes to safer medical decision-making.

---

## 📂 Dataset Files

- **`data synthetic.json`**: JSON Lines file where each line contains:
  - `"report"`: the full synthetic medical report  
  - `"summary"`: the corresponding summary  
  - `"omission"`: a boolean (`true`/`false`) indicating if factual content was omitted in the summary

---

## 📥 How to Use

Download the dataset using `huggingface_hub`:

```python
import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_json("hf://datasets/AchOk78/LLMgenerated_fictive_medical_report_and_summaries_with_omissions_label_Fr_Healthcare/synthetic data.json", lines=True)
````

---

## 📚 Dataset Creation

### Motivation

Due to GDPR and French healthcare privacy laws, the use of real medical data was not feasible. Anonymization also poses re-identification risks. Therefore, this dataset was synthetically generated to maintain realism without compromising privacy.

### Generation Process

1. Fifty real medical reports were anonymized by experts.
2. Each report was used as a prompt for GPT-4-32K to generate 15 diverse synthetic reports with instructions to vary family history, symptoms, dates, and complications.
3. Reports under 200 words were filtered out, resulting in 674 synthetic reports.
4. Each report was summarized and labeled by experts for omissions.

---

## ✅ Quality Assessment

* **Realism Test:** Two medical experts classified 100 reports (50 real, 50 synthetic). F1 scores of 45.4% and 54.7% suggest synthetic reports are indistinguishable from real ones.
* **Lexical Diversity:** PCA on CamemBERT embeddings shows synthetic and real reports cover similar lexical spaces.
* **Omission Label Validation:** Blind test on 90 summaries achieved 95% expert precision, confirming omission label reliability.

---

## 📄 Citation

If you use this dataset or method, please cite the associated paper (under review):

> **A Study on the Relevance of Generic Word Embeddings for Sentence Classification in Hepatic Surgery**
> Authors: \[To be added]
> Link: [https://arxiv.org/abs/your-paper-link](https://arxiv.org/abs/your-paper-link)

---

## 🔒 License

Released under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**:

* ✅ Academic and research use only
* ❌ Commercial use is **not allowed**
* 🔁 Derivative work must be shared under the same license
* 🏷 Proper attribution is required

For other uses, please contact the authors for explicit written permission.

---

## 🔧 Maintainer

**Author:** AchOk78
**Repo:** [https://github.com/achok7893/EmbedKDECheck\_OmissionsDection\_Dataset\_Fr\_Healthcare](https://github.com/achok7893/EmbedKDECheck_OmissionsDection_Dataset_Fr_Healthcare)
