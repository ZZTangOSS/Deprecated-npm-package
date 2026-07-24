# Artifact for "Deprecated but Not Abandoned: A Large-Scale Empirical Study on GDNPs"

This repository contains the supplementary materials, datasets, and replication scripts for the paper: **"Deprecated but Not Abandoned: A Large-Scale Empirical Study on Growing-User-Demand Deprecated NPM Packages"**. 

It provides all necessary components to reproduce the empirical study, including dataset construction, statistical modeling, security risk assessment, and topic modeling.

---

## 🗂 Repository Structure

The artifact is organized into five main directories corresponding to the dataset construction and the four research questions (RQs) addressed in our study:

```text
├── GDNP_dataset_construction/
│   ├── deprecated_npm_packages.csv
│   ├── GDNPs.csv
│   ├── Get_deprecated_NPM_packages.py
│   ├── Get_GDNPs.py
│   ├── Histogram_of_growth_rate.py
│   ├── names.json
│   └── Widely_used_deprecated.csv
├── RQ1_Repository-Level_Engagement_Analysis/
│   ├── GDNP_data.csv
│   ├── Non_GDNP_data.csv
│   └── RQ1_analysis.R
├── RQ2_Security_Risk_Assessment/
│   ├── Count_Security.py
│   ├── GDNPs_assessment.csv
│   ├── Non_GDNPs_assessment.csv
│   ├── OSV_Assessment.py
│   ├── RQ2_Figure.py
│   └── Widely_used_deprecated_NPM_packages...
├── RQ3_Survey_for_GDNP_Maintainers_and_Users/
│   ├── GDNP_maintainer_survey.md
│   ├── GDNP_user_survey.md
│   ├── Invitation_for_GDNP_maintainers.md
│   └── Invitation_for_GDNP_users.md
└── RQ4_Topic_Modeling/
    ├── Bertopic_Topic_Modeling.py
    ├── Clean_Before_Embedding.py
    ├── Embedding_qwen_0.6B.py
    └── Fetch_issues.py
```

---

## 📄 Detailed Description

### 1. `GDNP_dataset_construction/`
Contains scripts and intermediate datasets used to identify Growing-user-demand Deprecated NPM Packages (GDNPs).
* **`names.json`**: The list of package names retrieved from the NPM ecosystem.
* **`deprecated_npm_packages.csv` & `Widely_used_deprecated.csv`**: Datasets of all deprecated packages and the filtered widely-used subset (>10,000 monthly downloads).
* **`GDNPs.csv`**: The final dataset of the 864 identified GDNPs.
* **`Get_*.py`**: Python scripts for querying NPM registry metadata, filtering outliers, and applying log-transformed linear regression to identify growth trends.

### 2. `RQ1_Repository-Level_Engagement_Analysis/`
Contains the data and statistical modeling scripts used to analyze community engagement dynamics.
* **`*_data.csv`**: Time-series datasets recording monthly engagement metrics (stars, forks, issues, PRs) around the deprecation event.
* **`RQ1_analysis.R`**: The R script implementing the Regression Discontinuity Design (RDD) models to calculate fixed and random effects.

### 3. `RQ2_Security_Risk_Assessment/`
Contains scripts to quantify the vulnerability exposure introduced by the continued use of GDNPs.
* **`*_assessment.csv`**: The evaluation datasets containing the mapped security vulnerabilities for both package groups.
* **`OSV_Assessment.py`**: Script to map the identified packages to the Open Source Vulnerabilities (OSV) database.
* **`Count_Security.py`**: Script to compute the Risk Exposure Count based on CVSS severity levels and monthly download counts.

### 4. `RQ3_Survey_for_GDNP_Maintainers_and_Users/`
Contains the qualitative survey instruments used in our study.
* **`GDNP_*_survey.md`**: The complete structure and questions of the surveys distributed to maintainers and users.
* **`Invitation_for_*.md`**: The templates used for personalized email invitations.

### 5. `RQ4_Topic_Modeling/`
Contains the pipeline for collecting and clustering post-deprecation user discussions.
* **`Fetch_issues.py`**: Script utilizing the GitHub API to retrieve post-deprecation issues and comments.
* **`Clean_Before_Embedding.py`**: Data preprocessing script.
* **`Embedding_qwen_0.6B.py`**: Script for generating dense text embeddings.
* **`Bertopic_Topic_Modeling.py`**: The BERTopic implementation used to generate and cluster the semantic topics.

---

## 🔒 Ethical and Confidentiality

To adhere strictly to research ethics and protect participant privacy, this replication package does not contain the following raw data:

* **Raw Survey Responses**: We do not provide the raw answers from our 143 survey participants. Releasing this data would compromise the confidentiality guaranteed to our participants during the data collection phase.
* **Raw GitHub Issue Texts**: Although GitHub discussions are publicly accessible, we do not redistribute the raw discussion text. Sharing developers' discussions without their explicit consent may violate their privacy expectations. However, we provide the complete fetching and processing scripts in `RQ4_Topic_Modeling/` so researchers can independently replicate the data collection for follow-up studies.
