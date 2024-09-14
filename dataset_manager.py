import os
import pandas as pd
import pyarrow.parquet as pq
import json
from huggingface_hub import HfApi
import requests
from PIL import Image
from io import BytesIO


class DatasetManager:
    def __init__(self, dataset_name, dataset_dir="./dataset", parquet_file="001.parquet", num_rows=100,
                 num_files_to_download=None):
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.parquet_file_path = os.path.join(self.dataset_dir, parquet_file)
        self.df = None
        self.num_rows = num_rows  # Number of rows to load
        self.num_files_to_download = num_files_to_download  # Limit the number of files to download

        # Check and download the dataset if necessary
        self._download_dataset_if_needed()

        # Load a small portion of the dataset based on num_rows
        self._load_data_in_chunks()

    def _download_dataset_if_needed(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)
            print(f"Dataset not found in {self.dataset_dir}. Downloading...")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            hf = HfApi()

            # List all the files in the dataset
            files_metadata = hf.list_repo_files(repo_id=self.dataset_name, repo_type="dataset")

            # Limit the number of files based on num_files_to_download
            if self.num_files_to_download is not None:
                files_metadata = files_metadata[:self.num_files_to_download]

            # Download only the specified number of files
            for file in files_metadata:
                hf.download_file(repo_id=self.dataset_name, filename=file, repo_type="dataset",
                                 local_dir=self.dataset_dir)

            print(f"Download complete. Downloaded {len(files_metadata)} files.")
        else:
            print(f"Dataset found in {self.dataset_dir}.")

    def _load_data_in_chunks(self):
        if os.path.exists(self.parquet_file_path):
            print(f"Loading a portion of the dataset (up to {self.num_rows} rows)...")

            # Load only a small portion of the dataset using chunk reading
            parquet_file = pq.ParquetFile(self.parquet_file_path)

            # Reading the first N rows efficiently
            num_rows_loaded = 0
            batch_dataframes = []
            for batch in parquet_file.iter_batches(batch_size=self.num_rows):
                batch_df = batch.to_pandas()
                batch_dataframes.append(batch_df)
                num_rows_loaded += len(batch_df)
                if num_rows_loaded >= self.num_rows:
                    break

            self.df = pd.concat(batch_dataframes, ignore_index=True).head(self.num_rows)
            self._parse_json_columns()
            print(f"Dataset successfully loaded with {len(self.df)} samples.")
        else:
            print(f"File {self.parquet_file_path} not found.")

    def _parse_json_columns(self):
        # Parse JSON columns if they exist in the DataFrame
        if 'captions' in self.df.columns:
            self.df['captions'] = self.df['captions'].apply(json.loads)
        if 'metadata' in self.df.columns:
            self.df['metadata'] = self.df['metadata'].apply(json.loads)

    # Method to get the first few rows of the DataFrame
    def head(self, n=5):
        if self.df is not None:
            return self.df.head(n)
        else:
            print("No data available.")
            return None

    # Method to get the number of samples loaded
    def get_num_samples(self):
        if self.df is not None:
            return len(self.df)
        else:
            print("No data available.")
            return 0

    # Method to get specific columns
    def get_columns(self, columns):
        if self.df is not None:
            return self.df[columns].head()
        else:
            print("No data available.")
            return None

    def get_test(self) -> tuple[Image.Image, str, float]:
        """
        Generator that yields a tuple (url, caption, image) for each row.
        - url: The URL of the image.
        - caption: The caption associated with the image.
        - image: The loaded image (PIL Image object).
        """
        size = len(self.df)
        index = 0
        for _, row in self.df.iterrows():
            text = ' '.join([entry.get("text", "") for entry in row['metadata']['entries']])
            image = self._load_image_from_url(row['url'])
            index += 1
            yield image, text, index/size

    def _load_image_from_url(self, url: str) -> Image.Image:
        """
        Loads and returns an image from the given URL with a timeout of 2 seconds.
        If the loading takes more than 2 seconds, the image is skipped.
        """
        try:
            # Set a timeout of 2 seconds for the request
            response = requests.get(url, timeout=2)
            response.raise_for_status()  # Ensure the request was successful

            # Load the image from the response content
            img = Image.open(BytesIO(response.content))
            return img
        except requests.exceptions.Timeout:
            print(f"Timeout occurred while fetching image from {url}. Skipping...")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from {url}: {e}")

