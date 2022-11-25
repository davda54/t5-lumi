# coding=utf-8
import argparse
import numpy as np
import pickle
import gzip
import os
from smart_open import open

from tokenizers import Tokenizer


RANK = int(os.environ["SLURM_PROCID"])
print(RANK, flush=True)


class Processor:
    def __init__(self, ncc_path, c4_path, output_path, tokenizer) -> None:
        self.ncc_path = ncc_path
        self.c4_path = c4_path
        self.output_path = output_path
        self.tokenizer = tokenizer

    def load_and_tokenize(self):
        self.documents = [[]]

        total_tokens = 0
        for path in [self.ncc_path.format(RANK), self.c4_path.format(RANK)]:
            with gzip.open(path, "rt") as reader:
                for line in reader.readlines():
                    line = line.strip()

                    # Empty lines are used as document delimiters
                    if not line:
                        self.documents.append([])
                        continue

                    tokens = self.tokenizer.encode(line, is_pretokenized=False, add_special_tokens=False).ids
                    if len(tokens) > 0:
                        self.documents[-1].append(np.array(tokens))
                        total_tokens += len(tokens)

        # Remove empty documents
        self.documents = [document for document in self.documents if len(document) > 0]
        print(f"Loaded {total_tokens} tokens", flush=True)

    def save(self):
        for i in range(8):
            with gzip.open(self.output_path.format(RANK * 8 + i), mode='wb') as f:
                pickle.dump(self.documents[i::8], f, protocol=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", default="../data/wordpiece.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--ncc_path", default="../data/pretrain/NCC/segmented/{:03d}.txt.gz", type=str, help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--c4_path", default="../data/pretrain/c4/segmented/{:03d}.txt.gz", type=str)
    parser.add_argument("--output_path", default="../data/pretrain/tokenized/nor_ncc_{:04d}.pickle.gz", type=str, help="The output file where the model checkpoints will be written.")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.vocab_path)
    processor = Processor(args.ncc_path, args.c4_path, args.output_path, tokenizer)

    print("Loading...", flush=True)
    processor.load_and_tokenize()

    print("Saving...", flush=True)
    processor.save()
