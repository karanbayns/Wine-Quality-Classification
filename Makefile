# Aliases for directories and .csv files
RAW_DATA_PATH = data/raw/raw_data.csv
PROCESSED_DIR = data/processed/
TRAIN_DATA_PATH = $(PROCESSED_DIR)train_data.csv
TEST_DATA_PATH = $(PROCESSED_DIR)test_data.csv
EDA_RESULTS_DIR = results/eda/
ANALYSIS_RESULTS_DIR = results/models/

# all target to run all scripts in correct order
.PHONY: all
all : eda analyze

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

# clean target to delete all generated data and files
.PHONY: clean
clean :
	rm -f $(RAW_DATA_PATH)
	rm -rf $(PROCESSED_DIR)
	rm -rf $(EDA_RESULTS_DIR)
	rm -rf $(ANALYSIS_RESULTS_DIR)
	rm -rf src/__pycache__
	rm -rf data results