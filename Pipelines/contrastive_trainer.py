import torch
import torch.nn.functional as F
from transformers import Trainer
import pickle
import numpy as np


class ContrastiveTrainer(Trainer):
    def __init__(
            self,
            keyword_embeddings_path,
            label_to_cluster_map,
            contrastive_weight=0.3,
            temperature=0.07,
            **kwargs
    ):

        super().__init__(**kwargs)

        with open(keyword_embeddings_path, 'rb') as f:
            keyword_dict = pickle.load(f)

        sorted_keys = sorted(keyword_dict.keys())
        self.keyword_embeddings = torch.tensor(
            np.array([keyword_dict[k] for k in sorted_keys]),
            dtype=torch.float32
        )
        self.num_clusters = len(sorted_keys)

        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.label_to_cluster = label_to_cluster_map
        self.discovered_labels = set(label_to_cluster_map.keys())

        self.ce_losses = []
        self.contrastive_losses = []

        print(f"\n{'=' * 60}")
        print("ContrastiveTrainer Initialized")
        print('=' * 60)
        print(f"  Keyword embeddings: {self.keyword_embeddings.shape}")
        print(f"  Contrastive weight: {contrastive_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Label -> Cluster: {label_to_cluster_map}")
        print(f"  Loss = CE + {contrastive_weight} x Contrastive")
        print('=' * 60)

    def get_text_embeddings(self, model, inputs):
        base_model = None
        for attr in ['bert', 'roberta', 'xlm_roberta', 'electra', 'distilbert']:
            if hasattr(model, attr):
                base_model = getattr(model, attr)
                break

        if base_model is not None:
            base_outputs = base_model(**inputs)
            return base_outputs.last_hidden_state[:, 0, :]
        else:
            outputs = model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][:, 0, :]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        device = labels.device

        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = F.cross_entropy(logits, labels)

        text_embeddings = self.get_text_embeddings(model, inputs)

        discovered_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
        for label in self.discovered_labels:
            discovered_mask |= (labels == label)

        num_discovered = discovered_mask.sum().item()

        if num_discovered > 0:
            disc_embeddings = text_embeddings[discovered_mask]
            disc_labels = labels[discovered_mask]

            cluster_ids = torch.zeros(num_discovered, dtype=torch.long, device=device)
            for label_val, cluster_idx in self.label_to_cluster.items():
                cluster_ids[disc_labels == label_val] = cluster_idx

            text_norm = F.normalize(disc_embeddings, p=2, dim=1)
            kw_norm = F.normalize(self.keyword_embeddings.to(device), p=2, dim=1)

            similarity = torch.matmul(text_norm, kw_norm.T) / self.temperature
            contrastive_loss = F.cross_entropy(similarity, cluster_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        if self.state.global_step % 100 == 0:
            self.ce_losses.append(ce_loss.item())
            if num_discovered > 0:
                self.contrastive_losses.append(contrastive_loss.item())
            print(f"  [Step {self.state.global_step:5d}] CE: {ce_loss.item():.4f} | "
                  f"CL: {contrastive_loss.item():.4f} | Total: {total_loss.item():.4f} | "
                  f"Discovered: {num_discovered}/{len(labels)}")

        inputs["labels"] = labels

        return (total_loss, outputs) if return_outputs else total_loss

    def get_loss_summary(self):
        return {
            'avg_ce': np.mean(self.ce_losses) if self.ce_losses else 0,
            'avg_contrastive': np.mean(self.contrastive_losses) if self.contrastive_losses else 0,
        }
