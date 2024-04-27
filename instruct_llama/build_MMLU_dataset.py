from typing import Tuple, List, Mapping, Text, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import torch
import os
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import json


# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.models.model_lora import Transformer, LoraModelArgs
from instruct_llama.configs.sft_lora import config as cfg
from instruct_llama.models.tokenizer import Tokenizer
from instruct_llama.utils.logger import create_logger
from instruct_llama.utils.file_helper import (
    find_certain_files_under_dir,
    read_zipped_jsonl_file,
    read_json_file,
    read_jsonl_file,
    count_words,
)
from instruct_llama.core.prompt_builder import build_prompt_completion, Dialog

Metadata = Mapping[Text, Text]
"""
def process_mmlu_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048,  # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        'name': 'MMLU',
        'language': 'English',
        'home_page': 'https://huggingface.co/datasets/cais/mmlu',
    },
) -> None:
   # Process mmlu dataset and save the tokenized prompt:completion pairs to .pkl format.
   #
   # Here's an example format of prompt:completion pair before apply tokenization:
   # {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}

    

    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...')
        return

    if metadata is None:
        metadata = {}

    json_objs = read_jsonl_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing dolly dataset ...')
    datasets = []

    for item in json_objs:
        context = item['context'].strip()
        prompt = item['instruction'].strip()
        completion = item['response'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        if len(prompt_tokens) + len(completion_tokens) > max_seq_length:
            continue

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed dolly dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )

"""
TASKS = [
         'abstract_algebra',
        # 'anatomy',
        # 'astronomy',
        # 'business_ethics',
        # 'clinical_knowledge',
        # 'college_biology',
        # 'college_chemistry',
        # 'college_computer_science',
        # 'college_mathematics',
        # 'college_medicine',
        # 'college_physics',
        # 'computer_security',
        # 'conceptual_physics',
        # 'econometrics',
        # 'electrical_engineering',
        # 'elementary_mathematics',
        # 'formal_logic',
        # 'global_facts',
        # 'high_school_biology',
        # 'high_school_chemistry',
        # 'high_school_computer_science',
        # 'high_school_european_history',
        # 'high_school_geography',
        # 'high_school_government_and_politics',
        # 'high_school_macroeconomics',
        # 'high_school_mathematics',
        # 'high_school_microeconomics',
        # 'high_school_physics',
        # 'high_school_psychology',
        # 'high_school_statistics',
        # 'high_school_us_history',
        # 'high_school_world_history',
        # 'human_aging',
        # 'human_sexuality',
        # 'international_law',
        # 'jurisprudence',
        # 'logical_fallacies',
        # 'machine_learning',
        # 'management',
        # 'marketing',
        # 'medical_genetics',
        # 'miscellaneous',
        # 'moral_disputes',
        # 'moral_scenarios',
        # 'nutrition',
        # 'philosophy',
        # 'prehistory',
        # 'professional_accounting',
        # 'professional_law',
        # 'professional_medicine',
        # 'professional_psychology',
        # 'public_relations',
        # 'security_studies', 
        # 'sociology',
        # 'us_foreign_policy',
        # 'virology',
        #'world_religions'
    ]

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n 
#     return input_ids[-len(stop_ids)]

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        print(batch_input)
        print(f"yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def main(args):
    
    logger = create_logger()
    ckpt_dir = args.ckpt_dir
    param_size = args.param_size
    model_type = args.model_type
    tokenizer_dir = args.tokenizer_dir
    tokenizer = Tokenizer(cfg.tokenizer_file)


    results = {}
    output_filename = 'run_results_%s_%sb.json' % (model_type, param_size)
    
    logger.info('Initializing model and optimizer ...')

    torch.cuda.set_device('cuda:0')
    torch.cuda.empty_cache()

    model_args = LoraModelArgs.from_model_type(
        #r_MoE_k=cfg.r_MoE_k,
        model_type=cfg.model_type,
        # MoE configutations
        MoE=cfg.MoE,
        n_MoE_exp=cfg.n_MoE_exp,
        n_MoE_k=cfg.n_MoE_k,
        # LoRA configurations
        lora_r=cfg.lora_r,
        lora_scaling=cfg.lora_scaling,
        lora_dropout=cfg.lora_dropout,
        # LoRA trainable layers
        lora_attn_query=cfg.lora_attn_query,
        lora_attn_key=cfg.lora_attn_key,
        lora_attn_value=cfg.lora_attn_value,
        lora_attn_proj=cfg.lora_attn_proj,
        lora_attn_mlp=cfg.lora_attn_mlp,
        lora_head=cfg.lora_head,
        # Regular configurations
        head_type='lm_head',
        vocab_size=tokenizer.vocab_size,
        max_seq_len=cfg.max_seq_len,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
        head_dropout=cfg.head_dropout,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    model = Transformer(model_args)

    # Load model checkpoint using strict=False,
    # because there are missing keys due to LoRA weights not contained in checkpoint state
    if os.path.exists(ckpt_dir):
        logger.info(f'Loading pretrained checkpoint {ckpt_dir!r} ...')
        model_state = torch.load(ckpt_dir)
        model.load_state_dict(model_state, strict=False)
        del model_state
    """
    # try to convert the model to half precision, otherwise we can't even move the 7B model to a single RTX 3090
    for name, module in model.named_modules():
        module = module.to(dtype=compute_dtype)
    """

    model = model.to('cuda')

    start_time = time.time()
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        print(dev_df.shape)
        print(dev_df.size)
        print(test_df.shape)
        print(test_df.size)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, k)
            prompt = train_prompt + prompt_end
            """            while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)"""
            label = test_df.iloc[i, test_df.shape[1]-1]
            records.append({'prompt':prompt, 'answer':label})

        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str,  default='/home/yy/InstructLLaMA/instruct_llama/meta_checkpoints/llama-2-7b-chat/consolidated.pth')
    parser.add_argument('--param_size', type=str,  default= '7b-chat')
    parser.add_argument('--tokenizer_dir', type=str,  default= '/home/yy/InstructLLaMA/instruct_llama/meta_checkpoints/tokenizer.model')
    parser.add_argument('--model_type', type=str, default= 'llama2')
    parser.add_argument('--data_dir', type=str, default='/home/yy/InstructLLaMA/instruct_llama/datasets/mmlu/data')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()
    
    main(args)