# ReID Model

## Examples

=== "Fine-tuning `timm` model"

    ```python
    from trackers import ReIDModel
    from trackers.core.reid import get_market1501_dataset

    from torch.utils.data import DataLoader


    train_dataset, val_dataset = get_market1501_dataset(
        data_dir="datasets/reid/Market-1501-v15.09.15/bounding_box_train", split_ratio=0.9
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
    model.train(
        train_dataloader,
        epochs=10,
        validation_loader=val_dataloader,
        projection_dimension=len(train_dataset),
        freeze_backbone=True,
    )
    ```

=== "Load a fine-tuned checkpoint"

    ```python
    from trackers import ReIDModel

    model = ReIDModel.from_timm("checkpoints/reid_model_10.safetensors")
    ```

::: trackers.core.reid.dataset.base.TripletsDataset

::: trackers.core.reid.dataset.market_1501.get_market1501_dataset

::: trackers.core.reid.model.ReIDModel
