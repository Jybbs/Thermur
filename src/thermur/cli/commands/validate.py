"""
Validation command for the Thermur CLI.

This module provides the 'validate' command, which allows users to check
their system setup and configuration syntax without initiating a full
training run.
"""
from operator import itemgetter
from typer    import Context, Option


def validate(
    ctx              : Context,
    config_overrides : list[str] | None = Option(
        None,
        "--config", "-c",
        help = "Configuration overrides to validate"
    ),
):
    """
    ✅ Validate system setup and configuration without starting training.

    Performs comprehensive validation of the training environment including
    system requirements, configuration syntax, and integration status, providing
    a full report of any potential issues.
    """
    command = ValidateCommand(ctx)
    command.run(config_overrides)


class ValidateCommand:
    """
    Encapsulates the logic for the 'validate' command.

    This class provides a structured way to run all system and configuration
    validations, using components from the shared Typer context.
    """
    def __init__(self, ctx: Context):
        """
        Initializes the command with shared context components.

        Args:
            ctx: The Typer context, which holds the shared AppContext object
                 containing UI, system, and other core components.
        """
        self.cfg    = ctx.obj.cfg
        self.system = ctx.obj.system
        self.ui     = ctx.obj.ui

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status, displaying the results in a formatted table.
        """
        self.ui.print_section("System Information")

        info = self.system.get_system_info()
        self.ui.console.print(self.ui.create_system_table(info))
        self.ui.console.print()

        status, details = self.system.check_wandb_status()
        self.ui.console.print(f"[flock]🎨 wandb: {status} • {details}[/flock]")
        self.ui.console.print()

    def run(self, config_overrides: list[str] | None):
        """
        Executes the main validation workflow.

        Args:
            config_overrides: A list of Hydra configuration overrides to validate.
        """
        self.ui.print_header("System Validation")
        self._perform_system_validation()

        self.ui.print_section("Configuration Check")

        with self.ui.console.status(
            self.cfg.messages.status["validating_config"],
            spinner = "dots"
        ):
            issues = self.system.validate_config_overrides(
                overrides = config_overrides
            )

        msg_key = "config_issues" if issues else "config_passed"
        self.ui.print_message(
            message  = self.cfg.messages.validation[msg_key],
            msg_type = "warning" if issues else "success"
        )
        for i, issue in enumerate(issues, start=1):
            self.ui.console.print(f"  [warning]⚠️  {i}. {issue}[/warning]")

        self.ui.print_section("Integration Check")
        status, details = self.system.check_wandb_status()

        if "Not" in status:
            self.ui.print_message(f"wandb: {details}", "warning")
        else:
            self.ui.print_message(f"wandb: {details}", "success")

        self.ui.console.print()
        
        if any([issues, "Not" in status]):
            with_warn, review = itemgetter("with_warnings", "review_issues")(
                self.cfg.messages.validation
            )
            self.ui.print_message(with_warn, "warning")
            self.ui.print_message(review,    "tip")
        else:
            all_pass, ready = itemgetter("all_passed", "system_ready")(
                self.cfg.messages.validation
            )
            self.ui.print_message(all_pass, "success")
            self.ui.print_message(ready,    "success")