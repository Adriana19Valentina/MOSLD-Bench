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
            contrastive_weight=0.5,
            temperature=0.5,
            **kwargs
    ):
        super().__init__(**kwargs)

        with open(keyword_embeddings_path, 'rb') as f:
            keyword_dict = pickle.load(f)

        self.keyword_embeddings = torch.tensor(
            np.array([keyword_dict[i] for i in sorted(keyword_dict.keys())]),
            dtype=torch.float32
        )

        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        self.label_to_cluster = label_to_cluster_map
        self.discovered_labels = set(label_to_cluster_map.keys())

        print(f"ContrastiveTrainer initialized:")
        print(f"  Keyword embeddings shape: {self.keyword_embeddings.shape}")
        print(f"  Contrastive weight: {contrastive_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Label to cluster mapping: {label_to_cluster_map}")
        print(f"  Discovered labels: {self.discovered_labels}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: Cross-Entropy + Contrastive Loss
        """
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = F.cross_entropy(logits, labels)
        bert_outputs = model.bert(**inputs)
        text_embeddings = bert_outputs.last_hidden_state[:, 0, :]

        discovered_mask = torch.zeros_like(labels, dtype=torch.bool)
        for discovered_label in self.discovered_labels:
            discovered_mask |= (labels == discovered_label)

        if discovered_mask.sum() > 0:
            discovered_embeddings = text_embeddings[discovered_mask]
            discovered_labels = labels[discovered_mask]

            cluster_ids = torch.zeros_like(discovered_labels)
            for label_idx, cluster_idx in self.label_to_cluster.items():
                cluster_ids[discovered_labels == label_idx] = cluster_idx

            text_emb_norm = F.normalize(discovered_embeddings, dim=1)
            keyword_emb_norm = F.normalize(
                self.keyword_embeddings.to(text_emb_norm.device),
                dim=1
            )

            similarity = torch.matmul(text_emb_norm, keyword_emb_norm.T) / self.temperature

            contrastive_loss = F.cross_entropy(similarity, cluster_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=ce_loss.device)

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        if self.state.global_step % 100 == 0:
            num_discovered = discovered_mask.sum().item()
            if num_discovered > 0:
                print(f"  [Step {self.state.global_step}] "
                      f"CE: {ce_loss.item():.4f}, "
                      f"Contrastive: {contrastive_loss.item():.4f}, "
                      f"Total: {total_loss.item():.4f} "
                      f"(discovered: {num_discovered}/{len(labels)})")
            else:
                print(f"  [Step {self.state.global_step}] "
                      f"CE: {ce_loss.item():.4f}, "
                      f"Total: {total_loss.item():.4f} "
                      f"(no discovered samples)")

        return (total_loss, outputs) if return_outputs else total_loss