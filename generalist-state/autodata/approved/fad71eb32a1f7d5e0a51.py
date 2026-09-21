import os
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from datasets import load_dataset
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

num_cores = os.cpu_count()
script_dir = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
emb_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
batch_size = 3072
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tsne_perplexity = 100
emb_file = os.path.join(script_dir, "emb.pkl")
emb_2d_file = os.path.join(script_dir, "emb_2d.npy")
ratio = 1

print("1. Loading problems from datasets...")
dapo17k_problems = [str(p) if p is not None else "" for p in load_dataset("RyanYr/DAPO-Math-17k", split="train", trust_remote_code=True)["problem"]]
ds_preview_problems = [str(p) if p is not None else "" for p in load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train", trust_remote_code=True)["problem"]]
openr1_problems = [str(p) if p is not None else "" for p in load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", trust_remote_code=True)["problem"]]
openrs_problems = [str(p) if p is not None else "" for p in load_dataset("knoveleng/open-rs", split="train", trust_remote_code=True)["problem"]]
deepmath103k_problems = [str(p) if p is not None else "" for p in load_dataset("zwhe99/DeepMath-103K", split="train", trust_remote_code=True)["question"]]

orz_57k = requests.get("https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json").json()
orz_72k = requests.get("https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_72k_collection_extended.json").json()
orz_129k = orz_57k + orz_72k
orz129k_problems = [str(o[0]["value"]) for o in orz_129k]

def get_problem_from_still(sample):
    fields = sample["prompt"][0]["content"].split("User:")
    assert len(fields) == 2
    problem = fields[1].split("Assistant:")[0].strip()
    return problem
still_problems = [get_problem_from_still(sample) for sample in load_dataset("RUC-AIBOX/STILL-3-RL-90K", split="train")]

train_data_map = {
    "dapo17k": dapo17k_problems[:int(len(dapo17k_problems) * ratio)],
    "ds_preview": ds_preview_problems[:int(len(ds_preview_problems) * ratio)],
    "openr1": openr1_problems[:int(len(openr1_problems) * ratio)],
    "openrs": openrs_problems[:int(len(openrs_problems) * ratio)],
    "orz129k": orz129k_problems[:int(len(orz129k_problems) * ratio)],
    "still": still_problems[:int(len(still_problems) * ratio)],
    "deepmath103k": deepmath103k_problems[:int(len(deepmath103k_problems) * ratio)],
}
dataset_to_color = {
    "dapo17k": "#5f77a0",
    "openrs": "#5f77a0",
    "ds_preview": "#5f77a0",
    "orz129k": "#5f77a0",
    "openr1": "#5f77a0",
    "still": "#5f77a0",
    "deepmath103k": "#d1615e",
}
dataset_to_size = {
    "dapo17k": 5,
    "openrs": 5,
    "ds_preview": 5,
    "orz129k": 5,
    "openr1": 5,
    "still": 5,
    "deepmath103k": 5,
}
dataset_to_formal_name = {
    "dapo17k": "DAPO-17k",
    "openrs": "Open-RS",
    "ds_preview": "DSR-Preview",
    "orz129k": "ORZ-129k",
    "openr1": "Open-R1",
    "still": "STILL-3-RL",
    "deepmath103k": "DeepMath-103k",
}
dataset_names_ordered = list(train_data_map.keys())
dataset_formal_names_ordered = [dataset_to_formal_name[name] for name in dataset_names_ordered]
all_problems = []
for problems in train_data_map.values():
    all_problems.extend(problems)
num_problems = len(all_problems)

print("2. Encoding problems...")
if os.path.exists(emb_file):
    print(f"Loading embeddings from {emb_file}...")
    embeddings_tensor = torch.load(emb_file)
else:
    print(f"Encoding problems with {emb_model_name} on device {device}...")
    emb_model = SentenceTransformer(emb_model_name, device=device)
    if not all_problems:
        raise ValueError("No problems to encode. Check dataset loading.")
    embeddings_np = emb_model.encode(all_problems, batch_size=batch_size, show_progress_bar=True, device=device)
    embeddings_tensor = torch.from_numpy(embeddings_np)
    embeddings_tensor = nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    torch.save(embeddings_tensor, emb_file)
    print(f"Saved embeddings to {emb_file}")

embeddings_for_faiss = embeddings_tensor.cpu().numpy()
assert embeddings_for_faiss.shape[0] == num_problems, f"not match: {embeddings_for_faiss.shape[0]} != {num_problems}"

print("3. Determining the original dataset of each problem...")
problem_origins = []
for name in dataset_names_ordered:
    problems_list = train_data_map[name]
    problem_origins.extend([name] * len(problems_list))

problem_origins_formal = []
for name in dataset_names_ordered:
    problems_list = train_data_map[name]
    problem_origins_formal.extend([dataset_to_formal_name[name]] * len(problems_list))

print("4. Performing t-SNE dimensionality reduction...")
if not os.path.exists(emb_2d_file):
    tsne = TSNE(n_components=2, random_state=42, verbose=1, n_jobs=num_cores, perplexity=tsne_perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_for_faiss)
    np.save(emb_2d_file, embeddings_2d)
else:
    print(f"Loading embeddings from {emb_2d_file}...")
    embeddings_2d = np.load(emb_2d_file)

print("5. Plotting t-SNE for each dataset...")
for i, dataset_name in enumerate(dataset_names_ordered):
    tsne_plot_path = os.path.join(script_dir, f"tsne_plot_{dataset_name}.pdf")
    indices = [idx for idx, origin in enumerate(problem_origins) if origin == dataset_name]
    plt.figure(figsize=(16, 16))
    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        color=dataset_to_color[dataset_name],
        label=dataset_name,
        alpha=1.0,
        s=dataset_to_size[dataset_name]
    )
    plt.axis('off')
    plt.savefig(tsne_plot_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {tsne_plot_path}")
