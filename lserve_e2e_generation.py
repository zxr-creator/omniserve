# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }

import argparse
import json
from typing import List, Tuple
import random
import os
import glob

import datasets

import omniserve.utils.constants
from omniserve import EngineArgs, LLMEngine, SamplingParams
from omniserve.conversation import get_conv_template_name, get_conv_template

max_seq_len = omniserve.utils.constants.max_seq_len
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_PINK = "\033[45m"
RESET = "\033[0m"

random.seed(484)

def read_haystack_files(max_context_length, haystack_dir):
    context = ""
    while len(context) < max_context_length:
        if not os.path.exists(haystack_dir):
            raise ValueError(f"Directory {haystack_dir} does not exist")
        for file in glob.glob(f"{haystack_dir}/*.txt"):
            if not os.path.exists(file):
                raise ValueError(f"File {file} does not exist")
            with open(file, "r") as f:
                context += f.read()
    return context


def create_jsonl_prompts(
    dataset_path: str,
    max_tokens: int = 8192,
    stop_token_ids: List[int] = None,
    limit: int = None,
) -> List[Tuple[str, SamplingParams]]:
    """Load prompts from a JSONL file (one record per line, must have a `prompt` field).

    Mirrors the shape returned by `create_test_prompts`: a list of
    `(prompt, SamplingParams)` tuples. Used for AIME24-style benchmarks where
    every example carries an already-templated chat prompt.
    """
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset file {dataset_path} does not exist")

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        stop_token_ids=stop_token_ids if stop_token_ids is not None else [128001, 128009],
        max_tokens=max_tokens,
    )

    request_list: List[Tuple[str, SamplingParams]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt")
            if prompt is None:
                raise KeyError(f"Record in {dataset_path} missing 'prompt' field: keys={list(rec.keys())}")
            request_list.append((prompt, sampling_params))
            if limit is not None and len(request_list) >= limit:
                break

    print(f"{BG_PINK}Loaded {len(request_list)} prompts from {dataset_path}.{RESET}")
    return request_list


def create_test_prompts(conv_t, test_length=65536, test_depth=0.5, haystack_dir="./eval/needle/PaulGrahamEssays") -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    # NOTE: For the sake of simplicity, test_length is the string length, not token length.
    request_list = []
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=256
    )

    needle="\n\nRemember, the best two things to do in San Francisco are eating a sandwich and sitting in Dolores Park on a sunny day.\n\n"
    haystack_text=read_haystack_files(2*test_length, haystack_dir)

    needle_depth = int(test_length * test_depth)
    # print(f"needle_depth: {needle_depth}") 
    context = haystack_text[:needle_depth] + needle + haystack_text[needle_depth:test_length]

    retrieval_question = "what are the best two things to do in San Francisco?\n\nAnswer: The best two things to do in San Francisco are"
    test_prompt = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n\nQuestion: Based on the content of the book, {retrieval_question}"
    
    request_list.append((test_prompt, sampling_params))

    print(f"{BG_PINK}There are {len(request_list)} prompts to be processed.{RESET}")
    return request_list


def process_requests(engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            succeeded = engine.add_request(str(request_id), prompt, sampling_params)
            if succeeded:
                request_id += 1
        num_test_prompts = request_id

        if not test_prompts:
            break

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        max_context_length = 128
        max_gen_length = 384
        tot_length = (
            max_context_length + max_gen_length
        )  # Set the upper bound for (prompt + gen) length
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1
    finished = 0
    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        if len(requests_outputs) == 0:
            break
        print(
            BG_BLUE
            + "*" * 5
            + "Iteration %d (remaining req.s = %d)"
            % (iter, len(requests_outputs) + len(engine.scheduler.waiting))
            + "*" * 5
            + RESET
        )
        for request_output in requests_outputs:
            if request_output["finished"]:
                finished += 1
                print(
                    f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['text']}"
                )
        iter += 1
        if engine.ifb_mode == False:
            if iter == max_gen_length:  # Early exit
                for request_output in requests_outputs:
                    print(
                        f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['tokens']}"
                    )
                break
    assert num_test_prompts == finished
    print(f"{BG_PINK}{finished} requests are finished.{RESET}")


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    if getattr(args, "dataset_path", None):
        test_prompts = create_jsonl_prompts(
            dataset_path=args.dataset_path,
            max_tokens=args.dataset_max_tokens,
            limit=args.dataset_limit,
        )
    else:
        conversation_template = get_conv_template_name(args.model)
        test_prompts = create_test_prompts(
            conv_t=conversation_template
        )
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional path to a JSONL file with a 'prompt' field per record (e.g. AIME24). "
             "If set, replaces the default needle-in-haystack prompt.",
    )
    parser.add_argument(
        "--dataset-max-tokens",
        type=int,
        default=8192,
        help="max_tokens for SamplingParams when running a JSONL dataset (default: 8192).",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional cap on the number of JSONL records to run.",
    )
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
