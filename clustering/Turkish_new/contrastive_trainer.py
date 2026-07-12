"""
ContrastiveTrainer - Combined Cross-Entropy + Contrastive Loss

Loss = L_CE + λ × L_CL

Where L_CL pulls text embeddings toward their cluster's keyword anchor.
"""

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
        """
        Args:
            keyword_embeddings_path: Path to pickle with {cluster_id: embedding}
            label_to_cluster_map: {model_label_id: cluster_id}
            contrastive_weight: λ weight for contrastive loss
            temperature: τ for similarity scaling
        """
        super().__init__(**kwargs)

        # Load keyword embeddings (anchors)
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

        # Logging
        self.ce_losses = []
        self.contrastive_losses = []

        print(f"\n{'=' * 60}")
        print("ContrastiveTrainer Initialized")
        print('=' * 60)
        print(f"  Keyword embeddings: {self.keyword_embeddings.shape}")
        print(f"  Contrastive weight (λ): {contrastive_weight}")
        print(f"  Temperature (τ): {temperature}")
        print(f"  Label → Cluster: {label_to_cluster_map}")
        print(f"  Loss = CE + {contrastive_weight} × Contrastive")
        print('=' * 60)

    def get_text_embeddings(self, model, inputs):
        """Extract [CLS] embeddings - works for BERT, RoBERTa, etc."""
        # Find base model
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
        """Compute L = L_CE + λ × L_CL"""

        # CRITICAL: Remove labels from inputs to prevent model from computing its own loss
        labels = inputs.pop("labels")
        device = labels.device

        # Forward pass WITHOUT labels - model returns only logits
        outputs = model(**inputs)
        logits = outputs.logits

        # Cross-entropy loss (computed manually)
        ce_loss = F.cross_entropy(logits, labels)

        # Get text embeddings for contrastive loss
        text_embeddings = self.get_text_embeddings(model, inputs)

        # Find discovered class samples
        discovered_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
        for label in self.discovered_labels:
            discovered_mask |= (labels == label)

        num_discovered = discovered_mask.sum().item()

        if num_discovered > 0:
            disc_embeddings = text_embeddings[discovered_mask]
            disc_labels = labels[discovered_mask]

            # Map to cluster IDs
            cluster_ids = torch.zeros(num_discovered, dtype=torch.long, device=device)
            for label_val, cluster_idx in self.label_to_cluster.items():
                cluster_ids[disc_labels == label_val] = cluster_idx

            # Normalize
            text_norm = F.normalize(disc_embeddings, p=2, dim=1)
            kw_norm = F.normalize(self.keyword_embeddings.to(device), p=2, dim=1)

            # Similarity and contrastive loss
            similarity = torch.matmul(text_norm, kw_norm.T) / self.temperature
            contrastive_loss = F.cross_entropy(similarity, cluster_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # Combined loss
        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        # Logging
        if self.state.global_step % 100 == 0:
            self.ce_losses.append(ce_loss.item())
            if num_discovered > 0:
                self.contrastive_losses.append(contrastive_loss.item())
            print(f"  [Step {self.state.global_step:5d}] CE: {ce_loss.item():.4f} | "
                  f"CL: {contrastive_loss.item():.4f} | Total: {total_loss.item():.4f} | "
                  f"Discovered: {num_discovered}/{len(labels)}")

        # Put labels back for any downstream processing
        inputs["labels"] = labels

        return (total_loss, outputs) if return_outputs else total_loss

    def get_loss_summary(self):
        return {
            'avg_ce': np.mean(self.ce_losses) if self.ce_losses else 0,
            'avg_contrastive': np.mean(self.contrastive_losses) if self.contrastive_losses else 0,
        }