from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from scdvd.data import DiffusionDataModule
from scdvd.runtime import metadata_dict, record_run
from scdvd.mask_module import EmpiricalMaskModule


@hydra.main(version_base=None, config_path="configs", config_name="sciplex3")
def main(config: DictConfig) -> None:
    L.seed_everything(int(config.seed), workers=True)
    training = config.mask_training
    data = DiffusionDataModule(
        dataset=OmegaConf.to_container(config.dataset, resolve=True),
        batch_size=int(training.batch_size),
        num_workers=int(training.num_workers),
        pin_memory=bool(training.pin_memory),
        max_train_samples=training.max_train_samples,
        max_eval_samples=training.max_eval_samples,
        mixed_rows=False,
        validation_split=str(training.validation_split),
        validation_fraction=float(training.validation_fraction),
        validation_seed=int(training.validation_seed),
    )
    data.setup("fit")
    metadata = metadata_dict(data.metadata)
    model = EmpiricalMaskModule(
        model_config=OmegaConf.to_container(config.model, resolve=True),
        training_config=OmegaConf.to_container(training, resolve=True),
        metadata=metadata,
    )
    output_dir = Path(training.save_dir) / training.run_id
    record_run(output_dir, config, stage="train-mask", metadata=data.metadata,
               artifacts={"chemical_features": config.model.chemical_feature_path})
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="{epoch:03d}-{step}",
        monitor="val/mask_bce",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        accelerator=training.accelerator,
        devices=int(training.devices),
        max_epochs=int(training.max_epochs),
        precision=training.precision,
        callbacks=[checkpoint],
        log_every_n_steps=int(training.log_every_n_steps),
        default_root_dir=output_dir,
    )
    trainer.fit(model, datamodule=data)
    print(f"Best mask checkpoint: {checkpoint.best_model_path}")
    print(f"Best validation mask BCE: {checkpoint.best_model_score}")


if __name__ == "__main__":
    main()
