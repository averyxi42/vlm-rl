from transformers import AutoModelForCausalLM,AutoProcessor,AutoModelForImageTextToText
import torch

from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

processor = AutoProcessor.from_pretrained('google/gemma-3-4b-it')
# processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
device = 'cuda:0'

def preprocess_func(batch,idx,padding_length=1000):
    processed_features = processor(images=batch["images"], text=batch["prompt"], add_special_tokens=False,return_tensors="pt",padding="max_length",max_length=padding_length)
    for k in processed_features.keys():
        processed_features[k]=processed_features[k].squeeze()
    processed_features['data_idx'] = idx
    return processed_features

def get_features(dataset,padding_length=1000,num_proc=1):
    try:
        features = dataset.remove_columns(['completion'])
    except Exception as e:
        print(e)
        features = dataset
    features = features.map(
        maybe_apply_chat_template, fn_kwargs={"tokenizer": processor.tokenizer}, num_proc=num_proc
    )
    features = features.map(preprocess_func,with_indices=True,fn_kwargs={"padding_length":padding_length},num_proc=num_proc,remove_columns = features.column_names,)#,batched=True,batch_size=2)
    features.set_format('torch')
    return features

def infer_logprobs(model,features,token_ids = [236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819],batch_size=8,num_workers=2):
    all_logprobs = []
    # Instantiate the collator with your model's processor
    # data_collator = GenericDataCollator(processor.tokenizer)
    # The DataLoader will automatically batch the pre-processed tensors for you
    data_loader = DataLoader(features, batch_size=batch_size,num_workers=num_workers)#,collate_fn = data_collator)

    # The inference loop
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference"):
            # Move the entire batch of tensors to the GPU
            indices = batch['data_idx']
            # for k,v in batch.items():
            #     print(f"{k}: {v.shape}")
            # print(len(batch))
            if len(batch['input_ids'])>1:
                batch = {k: v.squeeze().to(device) for k, v in batch.items() if k!='data_idx'}
            else:
                batch = {k: v.to(device) for k, v in batch.items() if k!='data_idx'}
            # for k,v in batch.items():
            #     print(f"{k}: {v.shape}")
            # batch = batch.to(device)
            # Get model outputs
            outputs = model(**batch)
            # Calculate log probabilities for the last token and your chosen token_ids
            # The logic is the same as yours
            logprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
            target_logprobs = logprobs[:, token_ids]

            # Move results to CPU and store them
            all_logprobs.extend(target_logprobs.float().cpu().numpy())
    return all_logprobs

if __name__=="__main__":
    dataset = load_dataset("Aasdfip/popair_po_2_val_2_branches")['train'].select(range(100))
    features = get_features(dataset,num_proc=64)

    dtype = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained("Aasdfip/po3_branches_ckpt1440",torch_dtype=dtype).to(device)
    # model = torch.compile(model, mode="default")
    model.eval()
    
    logprobs = infer_logprobs(model,features)

    print(logprobs)
