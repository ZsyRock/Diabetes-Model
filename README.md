# Data Mining Analysis for Diabetes Prediction
This project aims to explore and understand patterns in the data through mining and analysis of diabetes prediction data sets, and build prediction models to help identify patients with diabetes. Below is a summary of the main steps and results of the project:
Although the performance of the model did not reach a high level on the current data set, by continuously trying different methods, including feature engineering and ensemble learning, we gained a deep understanding of the data and provided a basis for subsequent improvements.
Understand the main steps and decision-making process of data mining by reading the code and reports.
Tuned random forest models and tried gradient boosting tree models are provided for further optimization or comparison with other models.

# Next steps:
The technology of time series analysis can more comprehensively predict the incidence of diseases under different circumstances and improve the interpretability of the model.
Time series forecasting in healthcare is rapidly developing for its practical significance, with models playing a key role in predicting disease progression, estimating mortality, and assessing time-dependent risks. I summarized some well-known data sets and tools that demonstrate the huge potential of time series forecasting in advancing medical solutions.

## data set

In the medical field, several datasets stand out.

PTB -

The NYU dataset includes NYU Annotation, NYU Annotation-Manhattan, NYU Annotation-Brooklyn, unlabeled inpatient clinical records spanning 10 years.

Fine-tuning data sets, such as "NYU Read- mission", "NYU Mortality", etc., contain specific labels.

The UF Health Clinical Corpus is an aggregate of clinical narratives from the UF Health IDR, MIMIC-III corpus, PubMed collections, and Wikipedia articles, forming a corpus of more than 90 billion words.

i2b2-2012 focuses on temporal relationships in clinical narratives and provides discharge summaries with temporal information.

MIMIC-III is a public dataset containing ICD-9 codes, vital signs, medications, and other critical patient data from intensive care units.

CirCor DigiScope is the largest pediatric heart murmur dataset with heart murmur annotations.

## Model checkpoints and toolkits

A variety of model checkpoints and toolkits have emerged targeting healthcare applications.

NYUTron is a large-scale language model built around real-time structured/unstructured notes and electronic orders.

BioBERT is derived from BERT and optimized using biomedical datasets. It excels at pinpointing entities, discerning relationships, and answering queries.

ClinicalBERT uses the MIMIC-III dataset to adapt to the clinical domain, helping to predict patient outcomes, data anonymization and diagnosis.

Based on BERT, BlueBERT is trained using biomedical text data and is proficient in various biomedical NLP tasks.

Clairvoyance is an end-to-end pipeline for clinical decision support, enabling prediction, monitoring and personalized treatment planning.

ARL EEGModels is a collection of CNN models for EEG signal processing in Keras and TensorFlow.

DeepEEG is a deep learning library for EEG processing in Keras/TensorFlow, compatible with the MNE toolbox.
