"""
CLI configuration stores using hydra-zen.

These configs use hydra_zen.just() to wrap pure data structures for the CLI
framework. All models are defined in schemas.py with full documentation.
"""
from .schemas         import *
from config.utils.zen import thermur_make_all, thermur_store
from hydra_zen        import just

cli = thermur_store(group="cli")

cli(just(DisplayModel()),  name="display")
cli(just(DownloadModel()), name="download")
cli(just(MessagesModel()), name="messages")
cli(just(PromptsModel()),  name="prompts")
cli(just(GlobusSecrets()), name="secrets")
cli(just(SystemModel()),   name="system")
cli("${lightning.wandb}",  name="wandb")

thermur_make_all(cli)