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

=== "Load a fine-tuned checkpoint"

    ```python
    from trackers import ReIDModel
    from trackers.utils.torch_utils import load_safetensors_checkpoint

    # Load state dict and config from safetensors checkpoint
    state_dict, config = load_safetensors_checkpoint(
        "checkpoints/reid_model_10.safetensors"
    )

    # Create model architecture from config
    model = ReIDModel.from_timm(**config["model_metadata"])
    if config["projection_dimension"]:
        model.add_projection_layer(
            projection_dimension=config["projection_dimension"]
        )

    # Load state dict to the backbone model
    for k, v in state_dict.items():
        state_dict[k].to(model.device)
    model.backbone_model.load_state_dict(state_dict)
    ```

::: trackers.core.reid.model
