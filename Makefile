# Ensure main directories exist
init:
	mkdir -p RAW_DATA PROCESSED_DATA MODEL_EVALUATION BEST_MODEL

# ----------------------------------------------------------------------------------

# Data extraction targets
race_data: init
	python 10_race_data_extraction.py

quali_data: init
	python 11_quali_data_extraction.py

fp1_data: init
	python 12_fp1_data_extraction.py

fp2_data: init
	python 13_fp2_data_extraction.py

fp3_data: init
	python 14_fp3_data_extraction.py

# ----------------------------------------------------------------------------------

# Data aggregation targets
agg_race: extract_all
	python 20_race_data_aggregating.py

agg_quali: extract_all
	python 21_quali_data_aggregating.py

agg_fp1: extract_all
	python 22_fp1_data_aggregating.py

agg_fp2: extract_all
	python 23_fp2_data_aggregating.py

agg_fp3: extract_all
	python 24_fp3_data_aggregating.py

# ----------------------------------------------------------------------------------

# Analysis targets
winner_prediction: final_prep
	python 40_binary\(winner\)prediction_mode.py

position_prediction: final_prep
	python 41_position_prediction_mode.py

# ----------------------------------------------------------------------------------

extract_all: race_data quali_data fp1_data fp2_data fp3_data
	@echo "All data extraction completed successfully!"

final_prep: agg_race agg_quali agg_fp1 agg_fp2 agg_fp3
	python 30__final_data_prep.py
	@echo "Final data preparation completed successfully!"

analyze_all: winner_prediction position_prediction
	@echo "All analysis completed successfully!"


all: extract_all final_prep analyze_all
	@echo "Full data pipeline completed successfully!"

# Clean targets (DANGEROUS, DO NOT RUN)
# clean_data:
# 	rm -rf PROCESSED_DATA/*
# 	rm -rf RAW_DATA/*

# clean_models:
# 	rm -rf MODEL_EVALUATION/*
# 	rm -rf BEST_MODEL/*

# clean_all: clean_data clean_models
# 	@echo "All generated files cleaned successfully!"

# Phony targets
.PHONY: all extract_all analyze_all clean_data clean_models clean_all \
        race_data quali_data fp1_data fp2_data fp3_data init \
        agg_race agg_quali agg_fp1 agg_fp2 agg_fp3 final_prep \
        winner_prediction position_prediction
