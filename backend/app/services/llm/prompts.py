from pathlib import Path


PROMPTS_ROOT = Path(__file__).resolve().parents[4] / "prompts"


class PromptTemplate:
    def __init__(self, relative_path: str, version: str) -> None:
        self.relative_path = relative_path
        self.version = version
        self.path = PROMPTS_ROOT / relative_path

    def load(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt file missing: {self.path}")
        return self.path.read_text(encoding="utf-8")

    def render(self, **kwargs: object) -> str:
        template = self.load()
        return template.format(**kwargs)
