"""
this script is for processing redpajama full data on streaming mode and saving to megatron-format.
Different to `preprocess_data.py`, this script is mainly for performance. 
"""

import os
import time 
import argparse
from pprint import pprint
from typing import Dict, List

import nltk
import ray
import ray.data
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from indexed_dataset import MMapIndexedDatasetBuilder


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text
    

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Name of the dataset repository,e.g. togethercomputer/RedPajama-Data-1T"
    )
    group.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="for local mode, you need to provide local RedPajama dataset repository, e.g. /home/user/local"
    )
    group.add_argument(
        "--cache-dir",
        type=str,
        default='/root/.cache',
        help="Hugging Face Cache dir, where the hugging face dataset it stored"
    )
    group.add_argument(
        "--source",
        type=str,
        default='default',
        help="data source of the redpajama data, please choose from \
            ['arxiv', 'book', 'c4', 'common_crawl', 'github', 'stackexchange', 'wikipedia'] \
            by default the value is set to default"
    )
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group.add_argument(
        '--stream', 
        default=False, 
        action='store_true', 
        help="whether to load data from hugging face using streaming mode"
    )
    group.add_argument(
        "--load-batch-size", type=int, default=1000, help="only needed if you use streaming mode to read data from hugging face"
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--cpu-per-worker", type=int, default=1, help="Number of CPUs to use per worker"
    )
    
    args = parser.parse_args()   
    args.output_path = '/home/user/local'
    return args

def save_megatron(out_file, idx_file, docs):
    data_builder = MMapIndexedDatasetBuilder(out_file, dtype=np.uint16)

    for doc in docs['tokens']:    
        data_builder.add_item(np.array(doc, dtype=data_builder.dtype))
    
    data_builder.end_document()                        
    data_builder.finalize(idx_file)


def main():
    args = get_args()

    output_dir = f'{args.output_path}/{args.output_prefix}'
    data_source = args.source 
    eos_tokens = [0]
    keep_newlines = args.keep_newlines
    split_sentences = args.split_sentences
    cache_dir = args.cache_dir 
    ray.init(address='auto')
    pprint(ray.cluster_resources())
    num_nodes = len(ray.nodes())
    parallelism = num_nodes * args.cpu_per_worker

    def preprocess_megatron_sentence(batch: Dict[str, np.ndarray]) -> pd.DataFrame:
        
        task_id = ray.get_runtime_context().get_task_id()

        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)
        
        splitter = nltk.data.load("tokenizers/punkt/english.pickle")

        if keep_newlines:
            splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
        else:
            splitter = splitter 
        
        samples = batch['text'].tolist()
        
        ids = []
        lens = []
        
        for sample in samples:
            sample_ids = []
            for sentence in splitter.tokenize(sample):

                encoded = tokenizer(sentence, 
                                    truncation=False,
                                    padding=False)['input_ids']
                if len(encoded) > 0:
                    sample_ids = sample_ids + encoded
                
            sample_ids = sample_ids + eos_tokens
            ids.append(sample_ids)
            lens.append(len(sample))

        batch = pd.DataFrame({"tokens": ids, "length": lens})

        if not os.path.exists(f"{output_dir}/{data_source}"):
            os.makedirs(f"{output_dir}/{data_source}")

        out_file = f'{output_dir}/{data_source}/{task_id[:20]}.bin'
        idx_file = f'{output_dir}/{data_source}/{task_id[:20]}.idx'
    
        save_megatron(out_file, idx_file, batch)

        return pd.DataFrame({'task_id': [task_id]})
    

    def preprocess_megatron(batch: Dict[str, np.ndarray]) -> pd.DataFrame:
        
        task_id = ray.get_runtime_context().get_task_id()

        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b', use_fast=True)

        samples = batch['text'].tolist()
        
        ids = []
        lens = []
        
        for sample in samples:
            encoded = tokenizer(sample, 
                                truncation=False,
                                padding=False)  
            sample_id = encoded['input_ids'] + eos_tokens
            ids.append(sample_id)
            lens.append(len(sample))

        batch = pd.DataFrame({"tokens": ids, "length": lens})

        if not os.path.exists(f"{output_dir}/{data_source}"):
            os.makedirs(f"{output_dir}/{data_source}")

        out_file = f'{output_dir}/{data_source}/{task_id[:20]}.bin'
        idx_file = f'{output_dir}/{data_source}/{task_id[:20]}.idx'
    
        save_megatron(out_file, idx_file, batch)

        return pd.DataFrame({'task_id': [task_id]})
    
    if split_sentences:
        fn_name = 'preprocess_megatron_sentence'
    else:
        fn_name = 'preprocess_megatron'

    if args.stream:
        dataset = load_dataset(args.input, data_source, streaming=True)['train']

        idx = 1
        for rows in dataset.iter(batch_size=args.load_batch_size):
            print("-----------------------------")
            df = pd.DataFrame(rows)
            ray_dataset = ray.data.from_pandas(df)
            ray_dataset = ray_dataset.repartition(parallelism)

            tokenized_data = ray_dataset.map_batches(eval(fn_name), batch_format="numpy", batch_size=None)
            tokenized_data.materialize()

            if idx % 10 == 0:
                print(f"{idx} * {args.load_batch_size} samples were written to disk.")
            idx += 1
            print("============================")
    else:
        os.environ["RED_PAJAMA_DATA_DIR"] = args.data_dir 
        ds = load_dataset(args.input, args.source, cache_dir=cache_dir)['train']
        ray_dataset = ray.data.from_huggingface(ds)
        # create multiple data blocks  
        ray_dataset = ray_dataset.repartition(parallelism)

        tokenized_data = ray_dataset.map_batches(eval(fn_name), batch_format="numpy", batch_size=None)
        tokenized_data.materialize()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"\nthis script took {end-start}s.")

