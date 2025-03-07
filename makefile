up: 
	pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt
	python setup.python --dataset_type large
