from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from scdvd import scDVDModule
from scdvd.data import DiffusionDataModule
from scdvd.runtime import metadata_dict, record_run


@hydra.main(version_base=None, config_path="configs", config_name="sciplex3")
def main(config: DictConfig) -> None:
    L.seed_everything(int(config.seed), workers=True)
    data = DiffusionDataModule(
        dataset=OmegaConf.to_container(config.dataset, resolve=True),
        batch_size=int(config.training.batch_size),
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
        max_train_samples=config.training.max_train_samples,
        max_eval_samples=config.training.max_eval_samples,
        mixed_rows=False,
        validation_split=str(config.training.validation_split),
        validation_fraction=float(config.training.validation_fraction),
        validation_seed=int(config.training.validation_seed),
        population_set_size=int(config.model.population.set_size),
        regroup_each_epoch=bool(config.training.regroup_each_epoch),
        sampling_strategy=str(config.training.get("sampling_strategy", "sets")),
        partition_set_size=config.training.get("partition_set_size", None),
        drop_last_train=bool(config.training.get("drop_last_train", True)),
    )
    data.setup("fit")
    metadata = metadata_dict(data.metadata)
    model = scDVDModule(
        model_config=OmegaConf.to_container(config.model, resolve=True),
        training_config=OmegaConf.to_container(config.training, resolve=True),
        metadata=metadata,
    )

    output_dir = Path(config.training.save_dir) / config.training.run_id
    record_run(output_dir, config, stage="train", metadata=data.metadata,
               artifacts={"chemical_features": config.model.chemical_feature_path})
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="{epoch:03d}-{step}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=int(config.training.early_stopping_patience),
            min_delta=float(config.training.early_stopping_min_delta),
            check_finite=True,
            verbose=True,
        ),
    ]
    trainer = L.Trainer(
        accelerator=config.training.accelerator,
        devices=int(config.training.devices),
        max_epochs=int(config.training.max_epochs),
        precision=config.training.precision,
        gradient_clip_val=float(config.training.gradient_clip_val),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        callbacks=callbacks,
        log_every_n_steps=int(config.training.log_every_n_steps),
        check_val_every_n_epoch=int(config.training.check_val_every_n_epoch),
        default_root_dir=output_dir,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        reload_dataloaders_every_n_epochs=(
            1 if (bool(config.training.regroup_each_epoch)
                  or config.training.get("sampling_strategy", "sets") != "sets") else 0
        ),
    )
    trainer.fit(model, datamodule=data)
    print(f"Best validation checkpoint: {checkpoint.best_model_path}")
    print(f"Best validation loss: {checkpoint.best_model_score}")


if __name__ == "__main__":
    main()
