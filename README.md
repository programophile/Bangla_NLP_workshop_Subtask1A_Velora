# Bangla Hate Speech Detection: A Comprehensive Study with BanglaBERT Variants

This repository contains the code and resources for the ACL 2025 submission on Bangla Hate Speech Detection using various BanglaBERT-based models. The work focuses on multi-class classification of hate speech in Bangla text, participating in the BLP 2025 Hate Speech Subtask 1A.

## Abstract

Hate speech detection in low-resource languages like Bangla presents unique challenges due to limited annotated data and linguistic complexities. This study explores multiple variants of BanglaBERT, a pre-trained language model for Bangla, combined with advanced techniques such as hybrid architectures, data augmentation, focal loss, and regularization methods. We evaluate five different approaches on the BLP 2025 Hate Speech dataset, achieving competitive performance in identifying religious hate, sexism, political hate, profane language, abusive content, and non-hate speech.

## Repository Structure

The repository is organized as follows:

- `Code1(BanglaBERT-Hybrid (CNN-BiLSTM-Attn) )/` - Advanced hybrid model combining CNN, BiLSTM, and multi-head attention layers on top of BanglaBERT
- `Code2(Conservative BanglaBERT-Hybrid)/` - Conservative hybrid approach with simpler architecture for robust performance
- `code3(Balanced Augmented BanglaBERT)/` - Balanced data augmentation with focal loss and simplified classifier
- `Code4(Advanced BanglaBERT Fine- tune with CB-Focal & R-Drop)/` - Advanced fine-tuning with class-balanced focal loss and R-Drop regularization
- `Code5(Optimized BanglaBERT-Hybrid)/` - Optimized hybrid model with efficient architecture and training strategies
- `merge_dataset_code.ipynb` - Code for merging and preprocessing training datasets

Each code directory contains:
- Jupyter notebook (`.ipynb`) with the complete training and evaluation pipeline
- `predictor.py` - Python script for inference on test data
- Dataset files (`subtask_1A.tsv`, `blp25_hatespeech_subtask_1A_test_with_labels.tsv`)
- `format_checker/` - Directory for format validation scripts

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Datasets
- scikit-learn
- pandas
- numpy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/programophile/Bangla_NLP_workshop_Subtask1A_Velora.git
cd angla_NLP_workshop_Subtask1A_Velora
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn pandas numpy
```

3. Download the BanglaBERT model:
```bash
# The model will be automatically downloaded when running the notebooks
# csebuetnlp/banglabert
```

## Datasets

The experiments use the following datasets:

- **BLP 2025 Hate Speech Subtask 1A**: Official competition dataset containing Bangla text samples labeled for hate speech categories . Link : https://github.com/AridHasan/blp25_task1

-**Second dataset Link**: https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset
                                    
- **Merged Dataset**: Combination of BLP 2025 training data with additional Bangla hate speech data for improved training


### Data Format

Training/validation data format:
```
id	text	label
1000001	আমি বাংলায় গান গাই	None
1000002	এটা অসহ্য	Abusive
```

Test data format:
```
id	text
2000001	এই মানুষটা খারাপ
```

### Labels

- `None`: Non-hate speech
- `Religious Hate`: Hate speech targeting religious groups
- `Sexism`: Gender-based hate speech
- `Political Hate`: Politically motivated hate speech
- `Profane`: Profane or vulgar language
- `Abusive`: General abusive content

## Usage

### Training and Evaluation

Each model variant can be trained by running the corresponding Jupyter notebook:

1. **Code1 - Advanced Hybrid Model**:
```bash
cd "Code1(BanglaBERT-Hybrid (CNN-BiLSTM-Attn) )"
jupyter notebook code1.ipynb
```

2. **Code2 - Conservative Hybrid**:
```bash
cd "Code2(Conservative BanglaBERT-Hybrid)"
jupyter notebook code2.ipynb
```

3. **Code3 - Balanced Augmented**:
```bash
cd "code3(Balanced Augmented BanglaBERT)"
jupyter notebook code3.ipynb
```

4. **Code4 - Advanced Fine-tune**:
```bash
cd "Code4(Advanced BanglaBERT Fine- tune with CB-Focal & R-Drop)"
jupyter notebook code4.py  # For inference
```

5. **Code5 - Optimized Hybrid**:
```bash
cd "Code5(Optimized BanglaBERT-Hybrid)"
jupyter notebook code5.ipynb
```

### Data Preparation

Run the dataset merging script first:
```bash
jupyter notebook merge_dataset_code.ipynb
```

This creates the `merged_dataset.tsv` file used for training.

### Inference

For each model, use the `predictor.py` script to generate predictions on test data:

```bash
python predictor.py
```

The script will output predictions in the required format for submission.

## Models

### BanglaBERT Base Model
- **Model**: `csebuetnlp/banglabert`
- **Architecture**: BERT-based model pre-trained on Bangla corpus
- **Max Sequence Length**: 384 tokens

### Model Variants

1. **Hybrid CNN-BiLSTM-Attention**: Multi-layer CNN for feature extraction, BiLSTM for sequence modeling, multi-head attention for contextual understanding
2. **Conservative Hybrid**: Simplified hybrid architecture balancing complexity and performance
3. **Balanced Augmented**: Data augmentation techniques with focal loss for handling class imbalance
4. **CB-Focal & R-Drop**: Class-balanced focal loss with R-Drop regularization for improved generalization
5. **Optimized Hybrid**: Efficient architecture with optimized training strategies

## Results

### Performance Metrics

 Models                                     F1-Score
 Advanced BanglaBERT(Baseline)              0.7013
 Balanced Augmented BanglaBERT              0.7025
 BanglaBERT-Hybrid                          0.6954
 Optimized BanglaBERT-Hybrid                0.6886
 Conservative BanglaBERT-Hybrid             0.6793
 


### Key Findings

- Hybrid architectures combining CNN and RNN layers show improved performance over simple fine-tuning
- Data augmentation and focal loss help address class imbalance issues
- Regularization techniques like R-Drop improve model generalization
- Attention mechanisms enhance contextual understanding of hate speech patterns

## Citation

If you use this code or findings in your research, please cite our ACL 2025 paper:

```bibtex
@inproceedings{your-paper-2025,
  title={Bangla Hate Speech Detection: A Comprehensive Study with BanglaBERT Variants},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact [programophile@gmail.com].

## Acknowledgments

- BLP 2025 organizers for the hate speech dataset
- CSEBUET NLP Group for the BanglaBERT model
- Hugging Face for the Transformers library
