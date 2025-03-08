up: 
	pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
	pip3 install -r requirements.txt
	python3 setup.py --dataset_type large
