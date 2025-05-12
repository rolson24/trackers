# ReID Model

## Examples

=== "Fine-tuning `timm` model"

    ```python
    from trackers import ReIDModel
    from trackers.core.reid import get_market1501_dataset

    from torch.utils.data import DataLoader


    dataset = get_market1501_dataset(
        data_dir="datasets/reid/Market-1501-v15.09.15/bounding_box_train"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ReIDModel.from_timm(model_name="resnetv2_50.a1h_in1k")
    model.train(
        dataloader,
        epochs=10,
        projection_dimension=len(dataset),
        freeze_backbone=True,
    )
    ```

::: trackers.core.reid.model
