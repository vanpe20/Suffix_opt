# CKM Project Documentation

## Project Overview
1. Utilizing Sparse Autoencoders (SAEs) to identify latents indices that are primarily influenced by suffix additions (key latents are those with activation values above 0.5)
2. Visualizing and analyzing these latents' impact on specific words in the text using https://www.neuronpedia.org 
3. Based on the project at `/common/home/km1558/szr/CKM/rpo-main`, designing and training models
   - Corresponding paper: https://arxiv.org/abs/2401.17263

## Dataset Analysis

### 1. Suffix Dataset for Model Performance Enhancement
Location: `/common/users/km1558/szr_data/CKM/data/attackprompt`
- Contains 6 dataset categories
- Each category includes C3 and S1 suffix forms
- Baseline (no suffix) ground truth: `dev.txt`
- Corresponding prompt file: `prompt_ini.json`

### 2. Suffix-Enhanced Jailbreak Datasets (All created on AdvBench)
#### llm-attacks-main (GCG)
- Generated suffixes location: `/common/users/km1558/szr_data/CKM/data/jailbreak/result_output.txt`
- Project directory: `/common/users/km1558/szr_data/CKM/llm-attacks-main`
- Lightweight suffix generation code: `/common/home/km1558/szr/CKM/GCG.py`

#### AmpleGCG & AmpleGCG-plus
- Generated suffixes (official release): `/common/users/km1558/szr_data/CKM/data/jailbreak/AmpleGCG_Generated_Suffixes`
- Dataset documentation: `/common/users/km1558/szr_data/CKM/data/jailbreak/AmpleGCG_Generated_Suffixes/Data Organization.docx`
- Project directory: `/common/users/km1558/szr_data/CKM/AmpleGCG-main`

### 3. Key Latents Identified by SAEs
Location: `/common/home/km1558/szr/CKM/results`

#### Performance Analysis
SAE statistical method code:
- `/common/home/km1558/szr/CKM/sae_test/sae_GCG.py`
- `/common/home/km1558/szr/CKM/sae_test/sae_ampleGCG.py`

Results structure:
- `pure_data/`: Activation results for each dataset
- `suffix/`: Statistics of key latents (one file per information category)
- `together/`: Frequency statistics of commonly activated latents across different suffixes (latents with frequency ≥ 4 are considered key common influencers)

#### Jailbreak Analysis
SAE statistical method code:
- `/common/home/km1558/szr/CKM/sae_test/sae_performance.py`

Results structure:
- `GCG/AmpleGCG/AmpleGCG-plus.txt`: SAE activation results across three datasets
- `suffix.txt`: Key latent activation frequency statistics
- `detail.txt`: Activation value statistics for commonly activated key latents

#### Visualization
Location: `picture/`
- Visual representations of the statistical results

### 4. Key Latents Affecting Suffix Words (Based on Neuronpedia Analysis)
Location: `/common/home/km1558/szr/CKM/results/word.txt`
- Statistical analysis of latents that primarily influence words in suffixes
- Built upon the previous statistical foundation
- Results generated through Neuronpedia analysis

## Environment Requirements
- **RPO**: `conda activate rpo`
- **llm-attacks-main (GCG)**: `conda activate llact`
- **SAEs (this project)**: `conda activate ckm` 