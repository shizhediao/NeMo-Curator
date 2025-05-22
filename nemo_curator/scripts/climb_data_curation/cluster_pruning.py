# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import fasttext
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, read_data, write_to_disk
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper
from nemo_curator.modules.config import SemDedupConfig
import os
import inspect
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List
import re

# /lustre/fsw/portfolios/llmservice/users/sdiao/anaconda3/envs/nemo-curator/lib/python3.10/site-packages/nemo_curator/filters/classifier_filter.py
def preprocess_text(text):
    text = text.replace('\n', '<newline>')

    # Add spaces before and after punctuation
    text = re.sub(r'([.\!?,\'/()])', r' \1 ', text)
    # Convert to lowercase
    text = text.lower()
    # Merge multiple spaces into a single space
    text = ' '.join(text.split())
    return text

class FastTextModelManager:
    """
    FastText model manager for preloading and managing all FastText models
    """
    def __init__(self, model_path):
        """
        Initialize the model manager
        
        :param model_path: str, path to the FastText model
        """
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading FastText model...")
        self.model = fasttext.load_model(model_path)
        print(f"Successfully loaded model: {model_path}")   
    
    def get_scores(self, texts) -> List[float]:
        """
        Batch score a list of texts using the specified model
        
        :param texts: List of texts to score
        :return: List of scores
        """
        if not texts:
            return []
        texts = [preprocess_text(text) for text in texts]
        try:
            # Batch predict all texts
            predictions = self.model.predict(texts)
            # predictions returns a tuple (labels, probabilities)
            # labels is a list, each element corresponds to a predicted label for a text
            labels = predictions[0]
            # Convert labels to scores
            scores = [float(label[0].replace('__label__', '')) for label in labels]
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            scores = [None] * len(texts)
        
        return scores


def load_dataset(input_data_dir: str) -> DocumentDataset:
    files = list(get_all_files_paths_under(input_data_dir, keep_extensions="parquet"))
    raw_data = read_data(files, file_type="parquet", backend="pandas", add_filename=True, columns=['cosine_dist_to_cent', 'doc_id', 'file_name', 'l2_dist_to_cent', 'text']) # remove emb column
    return DocumentDataset(raw_data)

def main(args: argparse.Namespace) -> None:
    semdedup_config = SemDedupConfig.from_yaml(args.config_file)
    cache_dir = semdedup_config.cache_dir
    clustered_data_path = os.path.join(cache_dir, semdedup_config.clustering_save_loc, "embs_by_nearest_center")
    fasttext_model_path = "/lustre/fsw/portfolios/llmservice/users/sdiao/datatrove/examples/local/fasttext/askllm_likertscore_results_docs_1000000_fixed_fasttext_data/best_model_educational_value.bin"

    client = get_client(**ArgumentHelper.parse_client_args(args))  # noqa: F841

    subdirs = [os.path.join(clustered_data_path, d) for d in os.listdir(clustered_data_path)]
    for subdir in subdirs:
        cluster_id = subdir.split("/nearest_cent=")[-1]
        dataset = load_dataset(subdir)
        print(f"dataset.head(): {dataset.head()}")

        model_manager = FastTextModelManager(fasttext_model_path)
        texts = dataset.df['text'].compute().tolist()
        scores = model_manager.get_scores(texts)
        avg_score = sum(scores) / len(scores)

        if avg_score >= 1.0:
            filtered_output = os.path.join(cache_dir, semdedup_config.clustering_save_loc, "pruned_clusters.parquet", f"cluster_id={cluster_id}")
            write_to_disk(dataset.df, filtered_output, write_to_filename=True)
        else:
            continue

def attach_args() -> argparse.ArgumentParser:
    return ArgumentHelper.parse_semdedup_args(
        description=(
            "Extracts deduplicated data from the clustered embeddings of a collection of documents. "
            "This script requires that embeddings and clustering have been performed "
            "earlier using semdedup_extract_embeddings and semdedup_cluster_embeddings."
            "Input arguments include: "
            "--id-column for the the identifier in the dataset, "
            "--id-column-type for the data type of ID column, "
            "--config-file for the path to the semantic deduplication configuration file. "
            "Important configuration parameters include:"
            " cache_dir for the directory to store cache"
            " which_to_keep for specifying which duplicates to keep,"
            " sim_metric for the similarity metric for deduplication,"
            " and eps_to_extract for the epsilon value to extract deduplicated data."
        ),
    )


if __name__ == "__main__":
    # main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
    main(attach_args().parse_args())
