"""Canonical four-suite FinalVLA training entry point."""

from . import pi05, trainer


def main() -> None:
    # Importing pi05 installs the original AdamW and EMA checkpoint recipe.
    trainer.train_model()


if __name__ == "__main__":
    main()
