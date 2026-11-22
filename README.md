# EmbedKDECheck: Omission Detection in French Medical Summaries

A French dataset of **674 fictive medical reports** and their LLM-generated summaries, each labeled for omissions, plus **EmbedKDECheck**, a lightweight, reference-free method for detecting those omissions.

📄 Paper: [Détection des omissions dans les résumés médicaux générés par les grands modèles de langue](https://arxiv.org/abs/your-paper-link) (under review)

## Dataset

`synthetic data.json` is a JSON Lines file with one record per line:

| Field      | Type    | Description                              |
|------------|---------|------------------------------------------|
| `report`   | string  | Fictive medical report                   |
| `summary`  | string  | Summary of the report                    |
| `omission` | boolean | `true` if the summary omits key facts    |

```python
import pandas as pd

# Requires `huggingface-cli login`
df = pd.read_json(
    "hf://datasets/AchOk78/LLMgenerated_fictive_medical_report_and_summaries_with_omissions_label_Fr_Healthcare/synthetic data.json",
    lines=True,
)
```

All reports are fictive and contain no real patient data. Omission labels were assigned by medical experts and reached 95% precision in a blind test on 90 summaries.

## Method

EmbedKDECheck combines FastText and Word2Vec embeddings with unsupervised kernel density anomaly scoring. It needs no reference summary and no LLM calls, and it was compared against SelfCheckGPT, ChainPoll, G-Eval and GPTScore.

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, with attribution, and derivatives under the same license. For other uses, contact the authors.

## Citation

Please cite the paper above if you use this dataset or method.

Maintainer: [AchOk78](https://github.com/achok7893/EmbedKDECheck_OmissionsDection_Dataset_Fr_Healthcare)
