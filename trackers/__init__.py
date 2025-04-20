from trackers.core.sort.tracker import SORTTracker

__all__ = ["SORTTracker"]

try:
    from trackers.core.deepsort.feature_extractor import DeepSORTFeatureExtractor
    from trackers.core.deepsort.tracker import DeepSORTTracker

    __all__.extend(["DeepSORTFeatureExtractor", "DeepSORTTracker"])
except ImportError:
    print(
        "DeepSORT dependencies not installed. DeepSORT features will not be available. "
        "Please run `pip install trackers[deepsort]` and try again."
    )
    pass
