up: 
	pip install -r requirement.txt
	pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
	python setup.python --dataset_type large