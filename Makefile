.PHONY: all eda analyze reports clean

# Aliases for directories, .csv files, and reports
RAW_DATA_PATH = data/raw/raw_data.csv
PROCESSED_DIR = data/processed/
TRAIN_DATA_PATH = $(PROCESSED_DIR)train_data.csv
TEST_DATA_PATH = $(PROCESSED_DIR)test_data.csv
EDA_RESULTS_DIR = results/eda/
ANALYSIS_RESULTS_DIR = results/models/
REPORTS_DIR = reports/
REPORT_QMD = $(REPORTS_DIR)wine-quality.qmd
REPORT_HTML = $(REPORTS_DIR)wine-quality.html
REPORT_PDF = $(REPORTS_DIR)wine-quality.pdf

# all target to run all scripts in correct order
all : eda analyze reports

# Read .csv file (raw data) from the url
$(RAW_DATA_PATH): src/read_csv.py
	@mkdir -p $(dir $(RAW_DATA_PATH))
	python src/read_csv.py \
		https://raw.githubusercontent.com/prudhvinathreddymalla/Red-Wine-Dataset/refs/heads/master/winequality-red.csv \
	    $(RAW_DATA_PATH)

# Data processing and splitting into train and test sets
$(TRAIN_DATA_PATH) $(TEST_DATA_PATH): $(RAW_DATA_PATH) src/data_processing.py
	@mkdir -p $(PROCESSED_DIR)
	python -m src.data_processing \
		$(RAW_DATA_PATH) \
		$(PROCESSED_DIR)

# EDA of train data to create heatmap, histograms, and summary table
eda: $(TRAIN_DATA_PATH) src/eda.py
	@mkdir -p $(EDA_RESULTS_DIR)
	python src/eda.py \
		$(TRAIN_DATA_PATH) \
		$(EDA_RESULTS_DIR)

# Train the models using train and test data
analyze: $(TRAIN_DATA_PATH) $(TEST_DATA_PATH) src/analysis.py
	@mkdir -p $(ANALYSIS_RESULTS_DIR)
	python -m src.analysis \
		$(TRAIN_DATA_PATH) \
		$(TEST_DATA_PATH) \
		$(ANALYSIS_RESULTS_DIR)

# Render the reports in HTML and PDF format
reports: $(REPORT_HTML) $(REPORT_PDF)

$(REPORT_HTML): $(REPORT_QMD) eda analyze
	@mkdir -p $(REPORTS_DIR)
	quarto render $(REPORT_QMD) --to html

$(REPORT_PDF): $(REPORT_QMD) eda analyze
	@mkdir -p $(REPORTS_DIR)
	quarto render $(REPORT_QMD) --to pdf

# Clean target to delete all generated data and files
clean :
	rm -rf data results src/__pycache__
	rm -f $(REPORT_HTML) $(REPORT_PDF)