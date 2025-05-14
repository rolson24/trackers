# Re-Identification Models for Object Tracking

Re-identification (ReID) models are a key component in modern multi-object tracking systems. They extract appearance features from detected objects, enabling trackers like [DeepSORT](../deepsort/tracker.md) and ByteTrack to reliably associate identities across frames—even through occlusions, re-entries, and challenging scenarios.

## Installation

!!! example "Install ReID Dependencies"

    === "CPU"
        ```bash
        pip install "trackers[reid,cpu]"
        ```

    === "CUDA 11.8"
        ```bash
        pip install "trackers[reid,cu118]"
        ```

    === "CUDA 12.4"
        ```bash
        pip install "trackers[reid,cu124]"
        ```

    === "CUDA 12.6"
        ```bash
        pip install "trackers[reid,cu126]"
        ```

    === "ROCm 6.1"
        ```bash
        pip install "trackers[reid,rocm61]"
        ```

    === "ROCm 6.2.4"
        ```bash
        pip install "trackers[reid,rocm624]"
        ```

## Training a ReID Model for Person Re-Identification

We will use the [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)
dataset to train a ReID model for person re-identification.

```python
from trackers.core.reid import get_market1501_dataset

train_dataset, val_dataset = get_market1501_dataset(
    data_dir="datasets/reid/Market-1501-v15.09.15/bounding_box_train",
    split_ratio=0.9,
)
```

The `get_market1501_dataset` function returns a tuple of two datasets: `train_dataset` and `val_dataset` which are `TripletDataset`s that yield triplets of anchor, positive, and negative samples.

Next, we initialize a ReID model using the `from_timm` class method which initializes a pre-trained model from [timm](https://huggingface.co/docs/timm/en/index) which is able to extract pooled features corresponding to a given input image.

```python
from trackers import ReIDModel

model = ReIDModel.from_timm("resnetv2_50.a1h_in1k")
```

Now, we can train this model on the Market-1501 dataset.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model.train(
    train_dataloader,
    epochs=10,
    validation_loader=val_dataloader,
    learning_rate=5e-4,
    projection_dimension=len(train_dataset),
    freeze_backbone=True,
    checkpoint_interval=5,
)
```

!!! note
    By setting `projection_dimension=len(train_dataset)`, we are adding a projection layer to the model to project the features to a space of dimension equal to the number of unique identities in the dataset. We're also freezing the parameters in the pre-trained backbone model by setting `freeze_backbone=True`. Thus we only train the projection layer using Triplet Loss to learn the identity of each sample.

## Using Fine-tuned ReID Model with a Tracker

Now, we can use this fine-tuned ReID model checkpoint with a tracker to extract appearance features from detected objects and use them to associate identities across frames.

=== "DeepSORT"

    ```python hl_lines="2 5-6 13"
    import supervision as sv
    from trackers import DeepSORTTracker, ReIDModel
    from inference import get_model

    reid_model = ReIDModel.from_timm("logs/checkpoints/reid_model_10.safetensors")
    tracker = DeepSORTTracker(reid_model=reid_model)
    model = get_model(model_id="yolov11m-640")
    annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    def callback(frame, _):
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = tracker.update(detections, frame)
        return annotator.annotate(frame, detections, labels=detections.tracker_id)

    sv.process_video(
        source_path="input.mp4",
        target_path="output.mp4",
        callback=callback,
    )
    ```

=== "ByteTrack"

    Coming Soon!

## API Reference

::: trackers.core.reid.model.ReIDModel

::: trackers.core.reid.dataset.base.TripletsDataset

::: trackers.core.reid.dataset.market_1501.get_market1501_dataset
