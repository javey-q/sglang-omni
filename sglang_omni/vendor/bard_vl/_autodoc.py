"""Early config registration, ported from `nemo_automodel.components.models.utils`.

Upstream touches `CONFIG_MAPPING_NAMES` and `HARDCODED_CONFIG_FOR_MODELS` directly.
Both are transformers internals that moved between 4.57 and 5.x, so the writes are
best-effort here: `@auto_docstring` only needs them to resolve a config name, and a
missing mapping degrades the generated docstring rather than the model.
"""

from transformers import AutoConfig


def register_with_transformers_autodoc(model_type: str, config_cls) -> None:
    """Register custom configs early enough for `@auto_docstring` resolution."""
    config_name = config_cls.__name__
    model_types = {model_type, model_type.replace("_", "-")}

    AutoConfig.register(model_type, config_cls, exist_ok=True)

    for module_path, attr in (
        ("transformers.models.auto.configuration_auto", "CONFIG_MAPPING_NAMES"),
        ("transformers.utils.auto_docstring", "HARDCODED_CONFIG_FOR_MODELS"),
    ):
        try:
            module = __import__(module_path, fromlist=[attr])
            mapping = getattr(module, attr)
        except (ImportError, AttributeError):
            continue
        for key in model_types:
            mapping.setdefault(key, config_name)
