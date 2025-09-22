"""
Rich progress bar and model summary callbacks with Thermur styling.

This module provides customized PyTorch Lightning callbacks that integrate
with the Thermur UI design system for consistent visual output during training.
"""
from __future__                           import annotations
from config.types                         import TableColumn
from pytorch_lightning.callbacks          import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.progress import rich_progress as rp
from rich                                 import get_console
from thermur.cli.helpers                  import ThermurUI
from typing                               import TYPE_CHECKING

if TYPE_CHECKING:
    from config.cli.schemas import DisplayModel
    from pytorch_lightning  import LightningModule, Trainer
    from rich.progress      import ProgressColumn


class CallbackFactory:
    """
    Factory for creating Lightning callbacks with Thermur styling.

    Provides methods to instantiate customized callbacks that integrate
    with the Thermur UI design system for consistent visual output.
    """

    @staticmethod
    def create_model_summary(display: DisplayModel) -> ThermurModelSummary:
        """
        Create a model summary with Thermur's table styling.

        Uses the thermal color scheme and table formatting consistent
        with the rest of the CLI for the model architecture display.

        Args:
            display: Display configuration with styles and settings

        Returns:
            ThermurModelSummary configured with display settings
        """
        return ThermurModelSummary(display)

    @staticmethod
    def create_progress_bar(display: DisplayModel) -> ThermurProgressBar:
        """
        Create a progress bar using Thermur's display configuration.

        Uses the same thermal styling, progress bar length, and color scheme
        as the rest of the Thermur CLI for visual consistency.

        Args:
            display: Display configuration with styles and settings

        Returns:
            ThermurProgressBar configured with display settings
        """
        return ThermurProgressBar(display)


class ThermurModelSummary(RichModelSummary):
    """
    Custom RichModelSummary that uses Thermur's table styling.

    Overrides the summary display to use our thermal color scheme
    and table formatting conventions.
    """

    def __init__(self, display: DisplayModel):
        """
        Initialize with Thermur display configuration.

        Args:
            display: Display configuration with styles and settings
        """
        self.display = display
        self.ui      = ThermurUI(display)
        super().__init__()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when fit begins to display the model summary.

        Overrides to use our custom table styling for the model summary.

        Args:
            trainer   : PyTorch Lightning trainer instance
            pl_module : Lightning module being trained
        """
        if not self._max_depth:
            return

        summary = self._summary(trainer, pl_module)._get_summary_data()
        table   = self.ui.create_aligned_table(
            columns = [
                TableColumn("left",  self.display.styles['bright'],  "Layer",  25),
                TableColumn("left",  self.display.styles['flock'],   "Type",   30),
                TableColumn("right", self.display.styles['success'], "Params", 15)
            ],
            title = "Model Architecture"
        )

        [
            table.add_row(layer, type, "–"
            if (p := str(params).strip()) == "0" else p)
            for layer, type, params
            in zip(*(row[1] for row in summary[1:4]))
        ]

        get_console().print(table)


class ThermurProgressBar(RichProgressBar):
    """
    Custom RichProgressBar that uses Thermur's display configuration.

    Overrides column configuration to match the progress bar width
    and styling used throughout the Thermur CLI.
    """

    def __init__(self, display: DisplayModel):
        """
        Initialize with Thermur display configuration.

        Args:
            display: Display configuration with styles and bar settings
        """
        self.display = display

        super().__init__(
            console_kwargs = {"stderr": True},
            leave          = False,
            refresh_rate   = 20,
            theme          = rp.RichProgressBarTheme(
                batch_progress         = display.styles['muted'],
                description            = display.styles['bright'],
                metrics                = display.styles['flock'],
                metrics_format         = ".3e",
                metrics_text_delimiter = " • ",
                processing_speed       = display.styles['muted'],
                progress_bar           = display.styles['thermal'],
                progress_bar_finished  = display.styles['success'],
                progress_bar_pulse     = display.styles['warning'],
                time                   = display.styles['muted']
            )
        )

    def configure_columns(self, trainer: Trainer) -> list[ProgressColumn]:
        """
        Configure progress bar columns with Thermur styling.

        Overrides the default columns to use our configured bar width
        and maintain visual consistency.

        Args:
            trainer: PyTorch Lightning trainer instance

        Returns:
            List of configured Rich progress columns
        """
        return [
            rp.CustomBarColumn(
                bar_width      = self.display.progress_bar_length,
                complete_style = self.display.styles['thermal'],
                finished_style = self.display.styles['success'],
                pulse_style    = self.display.styles['thermal']
            ) if isinstance(c, rp.CustomBarColumn) else c
            for c in super().configure_columns(trainer)
        ]

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when training starts.

        Adds a line break before starting the progress bar display.

        Args:
            trainer   : PyTorch Lightning trainer instance
            pl_module : Lightning module being trained
        """
        get_console().print()
        super().on_train_start(trainer, pl_module)
