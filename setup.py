import os
import argparse
import urllib.request
import zipfile

from transformers import AutoTokenizer, AutoModel

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download pre-trained language model and MIND dataset."
    )

    parser.add_argument(
        "--model_name", type=str, nargs="?", help="Pre-trained language model from HuggingFace.", default="microsoft/MiniLM-L12-H384-uncased"
    )

    parser.add_argument(
        "--model_save_dir", type=str, nargs="?", help="Path to store downloaded model."
    )

    parser.add_argument(
        "--dataset_type", type=str, nargs="?", choices=["small", "large", "demo"], help="MIND dataset type.", default="small"
    )

    parser.add_argument(
        "--data_dir", type=str, nargs="?", help="Path to the downloaded zip files of MIND dataset.", default="./data"
    )

    parser.add_argument(
        "--zip_del", type=bool, nargs="?", help="Delete data zip file after extracting.", default=False
    )

    return parser.parse_args()

def extract_mind(dataset_type, root_folder, data_del):
    
    # mind_urls = {
    #     "small": {
    #         "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    #         "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
    #     },
    #     "large": {
    #         "train": "https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip",
    #         "dev": "https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip"
    #     },
    #     "demo": {
    #         "train": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip",
    #         "dev": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
    #     }
    # }

    print(f"Downloading MIND {dataset_type} dataset...")
    for split in ["train", "dev"]:
        zip_path = os.path.join(root_folder, f"MIND{dataset_type}_{split}.zip")
        extract_folder = os.path.join(root_folder, split)

        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {extract_folder}")
            os.makedirs(extract_folder, exist_ok=True)

            # Extract ZIP file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_folder)

            # Remove ZIP file after extraction
            if data_del:
                os.remove(zip_path)
            print(f"Extraction complete: {extract_folder}")
        else:
            print(f"Skipping {zip_path}: File not found. Please download and place it in {root_folder}.")

def main(args):
    # Download pre-trained languague model
    model_name = args.model_name
    model_save_dir = args.model_save_dir
    print(f"Downloading {model_name} model...")
    os.makedirs(model_save_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(model_save_dir)
    model.save_pretrained(model_save_dir)

    # Download MIND dataset
    data_dir = args.data_dir
    dataset_type = args.dataset_type
    data_del = args.zip_del

    extract_mind(dataset_type=dataset_type, root_folder=data_dir, data_del=data_del)

    print("Finished!")

if __name__ == "__main__":
    args = parse_arguments()
    if args.model_save_dir is None: 
        model_root = "./plm"
        model_dir = args.model_name.replace("/","_")
        args.model_save_dir = os.path.join(model_root, model_dir)

    main(args=args)
