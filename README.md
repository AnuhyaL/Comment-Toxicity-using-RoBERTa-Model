## About project ##

Toxic and harmful content on online platforms poses a serious threat to constructive discourse in the digital era. This paper presents an end-to-end, transformerbased automated comment toxicity detection model
utilizing RoBERTa, a language model pre-trained on large-scale text corpora. I fine-tuned the RoBERTa-base model on the Jigsaw Toxic Comment Classification Challenge dataset a widely used benchmark with six
toxicity categories: toxic, severe toxic, obscene, threat, insult, and identity hate. To address this problem, I formulated the task as a multi-label classification problem and applied binary cross-entropy loss with
sigmoid-based thresholding to predict multiple toxic traits simultaneously. The model achieved strong validation performance, with an accuracy of 92.97% and a macro F1-score of 0.68, indicating high effectiveness
in detecting nuanced toxic language. I performed an extensive evaluation including precision-recall analysis, loss curves, and a classification report to assess the model’s effectiveness across labels. This study
highlights the applicability of RoBERTa in industrial content moderation systems and suggests future research directions in fairness, bias mitigation, and crossplatform deployment for toxicity detection.

## Model File Information ##

The trained model file (`model.safetensors`) is **not included in this repository** because it exceeds GitHub's 100 MB file size limit.  
This file is stored locally for demonstration and testing purposes.  

If you need access to the trained model for evaluation or reproduction of results, refer to the shared Google Drive link (https://drive.google.com/file/d/1M8w8pXGKJnBn2Frpkq0Gx09fqQ4_K7cV/view?usp=drive_link).  
