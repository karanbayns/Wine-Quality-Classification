---
output:
  html_document: default
  pdf_document: default
---
# Wine-Quality-Classification Analysis
## Contributors
Aidan Hew

Karan Bains

SHUHANG LI

## Project Summary
Wine Quality Classification is a reproduciable project for classifying different red wine based on the quality. This project aims to investigate whether physicochemical properties can reliably predict wine quality using classification, which includes exploratory data analysis , model training, testing, and result visualization. The goal is to help understand the key physicochemical features affecting wine quality and build a model with good performance and generization.

## How to reproduce the Data Analysis
1. Clone the repository to your local machine
2. Create the virtual environment by using the following command line (if your laptop uses MacOS):
```bash
conda env create -f environment.yml
conda activate wine-quality
```
If your laptop does not use MacOS, use the following command line:
```bash
conda-lock install --name <environment_name>
conda activate <environment_name>
```
3. Run the following scripts in order:

    **Step 1: Download/Read Data (`read_csv.py`)**
    Reads the raw data and saves it to a local file.
    
    *Arguments*
    
    * `path_read`: URL or path to the input CSV.
    * `path_save`: **File path** (including filename) where the raw data should be saved.
    * `--delim`: (Optional) Delimiter of the input file (default: `,`).
    
    *Example*
    
    ```bash
    python src/read_csv.py <url_or_input_path> data/raw/raw_data.csv --delim ";"
    ```

    **Step 2: Process Data (`data_processing.py`)**
    Validates data schema, handles outliers/missing values, and splits data into train/test sets.
    
    *Arguments*
    
    * `path_read`: Path to the raw input CSV.
    * `path_save`: **Directory** where `train_data.csv` and `test_data.csv` will be saved.
    
    *Example*
    
    ```bash
    python src/data_processing.py data/raw/raw_data.csv data/processed/
    ```

    **Step 3: Exploratory Data Analysis (EDA) (`eda.py`)**
    Generates summary statistics, correlation heatmaps, and distribution plots from train data
    
    *Arguments*
    
    * `path_read`: Path to the training data CSV.
    * `path_save`: **Directory** where figures and tables will be saved.
    
    *Example*
    
    ```bash
    python src/eda.py data/processed/train_data.csv results/figures/
    ```

    **Step 4: Analysis (`analysis.py`)**
    Trains Logistic Regression, Decision Tree, and Random Forest models. Outputs performance metrics and ROC curves.
    
    *Arguments*
    
    * `path_train`: Path to the train data CSV.
    * `path_test`: Path to the test data CSV.
    * `path_save`: **Directory** where the model results (CSV and PNG) will be saved.
    
    *Example*
    
    ```bash
    python src/analysis.py data/processed/train_data.csv data/processed/test_data.csv results/models/
    ```
    **Important Note on Output Paths!!**

    * For `read_csv.py`, the `path_save` argument must be a **full file path** (e.g., `data/data.csv`).
    * For `data_processing.py`, `eda.py`, and `analysis.py`, the `path_save` argument must be a **directory** (e.g., `data/processed/`), 
      because the filenames are hardcoded within the scripts.

## The way to use the container image
1. Since we provide the 'docker-compose.yml' file, use the command line 'docker compose up -d', it will create a container and you will see the similar result below.
<img width="673" height="59" alt="截屏2025-11-29 上午11 47 18" src="https://github.com/user-attachments/assets/b27dd873-45ac-4a41-93d1-5342a636e271" />

2. Use the command line 'docker ps' to see the status of the container we created
<img width="1511" height="72" alt="截屏2025-11-29 上午11 48 22" src="https://github.com/user-attachments/assets/51f54202-82a7-46ff-b8b6-66447fc48265" />

3. Use the command line 'docker logs wine-quality-classification-analysis-env-1' ('wine-quality-classification-analysis-env-1' is the name of the container can be find in the 'docker ps' result)
<img width="706" height="77" alt="截屏2025-11-29 上午11 50 19" src="https://github.com/user-attachments/assets/66c20388-559c-460e-ad37-51e67b87e337" />

4. The result of step3 include URLs, use the second one, and open it in the broswer, and you will see the whole project opend in the Jupyter Lab.
5. Now you can run the code to reproduce the anaylsis process.

## The way to update container image
1. Stop and remove the original one by using 'docker compose down'
2. Pull the latest version of the images defined in `docker-compose.yml` by using 'docker compose pull'
3. And just follow the process we mentioned in the 'The way to use the container image' so that you can use the updated container image.



## Dependencies
  - pandas
  - scikit-learn
  - jupyter
  - python=3
  - numpy
  - matplotlib
  - altair

## Name of the license
The project is licensed under the MIT License and CC BY-NC-ND 4.0 license. The detail is in LICENSE.md.
