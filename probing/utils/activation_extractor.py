import os
import torch
from tqdm import tqdm

def extract_and_save_activations(
    model,
    saes,
    data_loader,
    device,
    save_dir,
    dataset_name,
    split,
    language,
    logger,
    skip_existing: bool = True,
):
    """
    Extract SAE pre-activations for each layer and save in a single file:
        save_dir/sae_acts/{dataset_name}/{language}-{split}.pt

    The saved dict structure:
    {
        "layer_name": tensor [num_batches * batch_size, seq_len, hidden_dim],
        ...
    }
    """

    out_dir = os.path.join(save_dir, "sae_acts", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{language}-{split}.pt")

    # ---- Skip if already exists ----
    if skip_existing and os.path.exists(out_path):
        logger.info(f"⚠️ Skipping extraction: file already exists at {out_path}")
        return

    acts_dict = {}

    for layer_index, (layer_name, sae_model) in enumerate(saes.items()):
        logger.info(f"\n[Layer {layer_index}] Processing {layer_name}...")
        sae_model.to(device)
        layer_outputs = []

        for batch in tqdm(data_loader, desc=f"Layer {layer_name}"):
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.hidden_states  # list of hidden states

            hidden = hidden_states[layer_index + 1]
            sae_latents = sae_model.encode(hidden)

            # Save **all neuron activations** (pre-activations)
            sae_preacts = sae_latents.pre_acts.cpu()
            layer_outputs.append(sae_preacts)

            del outputs, hidden, sae_latents, sae_preacts, input_ids, attention_mask
            torch.cuda.empty_cache()

        acts_dict[layer_name] = torch.cat(layer_outputs, dim=0)
        logger.info(f" Collected {layer_name} activations | shape = {acts_dict[layer_name].shape}")

        del layer_outputs
        torch.cuda.empty_cache()

    # ---- Save results ----
    torch.save(acts_dict, out_path)
    logger.info(f"\n✅ Saved all activations to: {out_path}")
