"""Train the Regulus heterogeneous graph encoder."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch

from regulus.graph.assets import DEFAULT_MANIFEST_NAME, resolve_graph_asset_dir
from regulus.graph.build import HeteroDataBuilder
from regulus.graph.hgt_trainer import Trainer
from regulus.graph.losses import HeteroLoss
from regulus.graph.model import HeteroRegulatorNet
from regulus.graph.schema import MESSAGE_PASSING_EDGE_TYPES, TF_CFO_LLM
from regulus.utils.config import load_config
from regulus.utils.reproducibility import set_random_seeds

logger = logging.getLogger(__name__)


def run_graph_training(
    config_path: str = "configs/graph_config.yaml",
    *,
    graph_asset_dir: Optional[str] = None,
    output_dir: str = "outputs",
    output_suffix: str = "graph",
    resume: Optional[str] = None,
    seed: int = 42,
    gpu: int = 0,
) -> Path:
    """Train HGT, export node embeddings, and return the final checkpoint path."""
    os.makedirs(output_dir, exist_ok=True)
    config = load_config(config_path)
    set_random_seeds(config.get("seeds", {}).get("seed", seed))

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    data_config = config.get("data", {})
    asset_dir = resolve_graph_asset_dir(config, graph_asset_dir)
    builder = HeteroDataBuilder(
        asset_dir,
        manifest_name=data_config.get("manifest", DEFAULT_MANIFEST_NAME),
        validate_hashes=data_config.get("validate_hashes", True),
    )
    stats = builder.get_data_statistics()
    train_data = builder.build_hetero_data(split_type=None, data_splitter=None)
    val_data = builder.build_hetero_data(split_type=None, data_splitter=None)

    if TF_CFO_LLM in train_data.edge_types:
        edge_count = train_data[TF_CFO_LLM].edge_index.shape[1]
        logger.info("TF->CFO LLM evidence edges: %s", f"{edge_count:,}")

    model_config = config["model"]
    node_types = model_config.get("node_types", ["tf", "gene", "celltype", "go"])
    edge_types = [
        tuple(edge_type)
        for edge_type in model_config.get("edge_types", MESSAGE_PASSING_EDGE_TYPES)
    ]
    model = HeteroRegulatorNet(
        node_types=node_types,
        edge_types=edge_types,
        node_feature_dims=builder.get_feature_dims(),
        hidden_dim=model_config["hidden_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config.get("dropout", 0.1),
        use_reconstruction=model_config.get("use_reconstruction", True),
        use_gradient_checkpointing=model_config.get("use_gradient_checkpointing", True),
    )

    training_config = config["training"]
    trainer = Trainer(
        model=model,
        loss_fn=HeteroLoss(**config["loss"]),
        device=device,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 1e-5),
        grad_clip_norm=training_config.get("grad_clip_norm", 1.0),
        use_mixed_precision=training_config.get("use_mixed_precision", True),
        patience=training_config.get("patience", 10),
        min_delta=training_config.get("min_delta", 1e-4),
        save_dir=os.path.join(output_dir, f"checkpoints_{output_suffix}"),
        edge_mask_ratio=training_config.get("edge_mask_ratio", 0.1),
        recon_mask_ratio=training_config.get("recon_mask_ratio", 0.3),
    )
    if resume:
        trainer.load_checkpoint(resume)

    history = trainer.fit(
        train_data,
        val_data,
        epochs=training_config["epochs"],
        log_interval=training_config.get("log_interval", 10),
    )
    trainer.plot_training_curves(
        os.path.join(output_dir, f"training_curves_{output_suffix}.png")
    )
    trainer.save_training_history(
        os.path.join(output_dir, f"training_history_{output_suffix}.json")
    )

    model.eval()
    with torch.no_grad():
        full_data = builder.build_hetero_data(split_type=None, data_splitter=None).to(device)
        x_dict = {
            node_type: full_data[node_type].x
            for node_type in model.node_types
            if node_type in full_data.node_types
        }
        edge_index_dict = {
            edge_type: full_data[edge_type].edge_index
            for edge_type in model.edge_types
            if edge_type in full_data.edge_types
        }
        embeddings = {
            node_type: values.cpu().numpy()
            for node_type, values in model.encode(x_dict, edge_index_dict).items()
        }

    embeddings_dir = Path(output_dir) / f"embeddings_{output_suffix}"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for node_type, values in embeddings.items():
        np.save(embeddings_dir / f"{node_type}_embeddings.npy", values)
    with open(embeddings_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "embedding_dim": model.hidden_dim,
                "num_nodes": {key: value.shape[0] for key, value in embeddings.items()},
                "node_types": list(embeddings),
            },
            handle,
            indent=2,
        )

    final_path = Path(output_dir) / f"final_model_{output_suffix}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stats": stats,
            "history": history,
            "node_embeddings": embeddings,
        },
        final_path,
    )
    return final_path


def run_graph_training_from_args(args: Union[SimpleNamespace, object]) -> Path:
    """Run graph training from an argparse-compatible namespace."""
    return run_graph_training(
        config_path=getattr(args, "config", "configs/graph_config.yaml"),
        graph_asset_dir=getattr(
            args,
            "graph_asset_dir",
            getattr(args, "data_dir", None),
        ),
        output_dir=getattr(args, "output_dir", "outputs"),
        output_suffix=getattr(args, "output_suffix", "graph"),
        resume=getattr(args, "resume", None),
        seed=getattr(args, "seed", 42),
        gpu=getattr(args, "gpu", 0),
    )
