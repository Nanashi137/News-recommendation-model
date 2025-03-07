import os
import argparse
import urllib.request
import zipfile

from transformers import AutoTokenizer, AutoModel

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download pre-trained language model and MIND dataset"
    )

    parser.add_argument(
        "--model_name", type=str, nargs="?", help="Pre-trained language model from HuggingFace", default="microsoft/MiniLM-L12-H384-uncased"
    )

    parser.add_argument(
        "--model_save_dir", type=str, nargs="?", help="Path to store downloaded model"
    )

    parser.add_argument(
        --"dataset_type", type=str, nargs="?", choices=["small", "large", "demo"], help="MIND dataset type", default="small"
    )

    parser.add_argument(
        "--data_dir", type=str, nargs="?", help="Path to save downloaded MIND dataset", default="./data"
    )

    return parser.parse_args()

def download_mind(dataset_type, root_folder):
    # Mapping dataset type to URLs
    mind_urls = {
        "small": {
            "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
        },
        "large": {
            "train": "https://mind201910.blob.core.windows.net/release/MINDlarge_train.zip",
            "dev": "https://mind201910.blob.core.windows.net/release/MINDlarge_dev.zip"
        },
        "demo": {
            "train": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip",
            "dev": "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
        }
    }


    print(f"Downloading MIND {dataset_type} dataset...")
    os.makedirs(root_folder, exist_ok=True)

    
    for split in ["train", "dev"]:
        url = mind_urls[dataset_type][split]
        zip_filename = os.path.join(root_folder, f"MIND{dataset_type}_{split}.zip")
        extract_folder = os.path.join(root_folder, split)

        
        os.makedirs(extract_folder, exist_ok=True)

        print(f"Downloading {split} set...")
        urllib.request.urlretrieve(url, zip_filename)
        print(f"Downloaded: {zip_filename}")

        print(f"Extracting {zip_filename} to {extract_folder}...")
        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        os.remove(zip_filename)
        print(f"Removed {zip_filename}")

    print("Download complete!")

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
    dataset_type = args.data_type

    download_mind(dataset_type=dataset_type, root_folder=data_dir)

if __name__ == "__main__":
    args = parse_arguments()
    if args.model_save_dir is None: 
        model_root = "./plm"
        model_dir = args.model_name.replace("/","_")
        args.model_save_dir = os.path.join(model_root, model_dir)

    main(args=args)