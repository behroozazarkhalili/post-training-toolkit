
from post_training_toolkit.integrations.trl import DiagnosticsCallback, TrainerType

# Backward-compatible aliases following TRL naming conventions (XxxCallback)
PTTCallback = DiagnosticsCallback
PostTrainingCallback = DiagnosticsCallback


def __getattr__(name: str):
    if name == "TransformersCallback":
        from post_training_toolkit.integrations.transformers import TransformersCallback
        return TransformersCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiagnosticsCallback",
    "PTTCallback",
    "PostTrainingCallback",
    "TrainerType",
    "TransformersCallback",
]
