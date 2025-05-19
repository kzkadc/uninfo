from collections.abc import Sequence
import json


def create_prompt_texts(class_texts: Sequence[str], template: str | None = None,
                        use_ensemble: str | None = None) -> list[list[str]]:
    match use_ensemble:
        case None:
            assert template is not None, "template must not be None when not using ensemble"
            templates = [template]
        case "imagenet":
            with open("prompt/imagenet_ensemble_prompts.json", "r", encoding="utf-8") as f:
                templates = json.load(f)
        case "cifar":
            with open("prompt/cifar_ensemble_prompts.json", "r", encoding="utf-8") as f:
                templates = json.load(f)

        case t:
            raise ValueError(f"Invalid template: {t!r}")

    template_texts = [
        [t.format(c) for t in templates]
        for c in class_texts
    ]
    return template_texts
