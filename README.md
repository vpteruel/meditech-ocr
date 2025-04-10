# meditech-ocr


python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 generate_dataset.py -d -q 10 -debug



TESSDATA_PREFIX=../tessdata make training MODEL_NAME=Meditech START_MODEL=eng TESSDATA=../tessdata MAX_ITERATIONS=1