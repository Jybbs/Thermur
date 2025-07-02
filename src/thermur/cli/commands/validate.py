"""
Validation command for the Thermur CLI.

This module provides the 'validate' command, which allows users to check
their system setup and configuration syntax without initiating a full
training run.
"""
from typer import Context, Option


def validate(
    ctx              : Context,
    config_overrides : list[str] | None = Option(
        default     = None,
        param_decls = ["--config", "-c"],
        help        = "Configuration overrides to validate"
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
            ctx : The Typer context, which holds the shared AppContext object
                  containing UI, system, and other core components.
        """
        self.config = ctx.obj.config
        self.system = ctx.obj.system
        self.ui     = ctx.obj.ui

    def _perform_system_validation(self):
        """
        Performs comprehensive system validation checks.

        This helper validates hardware capabilities, software versions, and
        integration status, displaying the results in a formatted table.
        """
        self.ui.print_major_section("System Information")

        info = self.system.get_system_info(self.config.wandb_integration)
        self.ui.console.print(self.ui.create_system_table(info))
        self.ui.console.print()

        status, details = self.system.check_wandb_status(self.config)
        self.ui.console.print(f"[flock]🎨 wandb: {status} • {details}[/flock]")
        self.ui.console.print()

    def run(self, config_overrides: list[str] | None):
        """
        Executes the main validation workflow.

        Args:
            config_overrides : A list of Hydra configuration overrides to validate.
        """
        self.ui.print_header("System Validation")
        self._perform_system_validation()

        self.ui.print_major_section("Configuration Check")

        with self.ui.console.status(
            self.config.messages.status["validating_config"],
            spinner = "dots"
        ):
            issues = self.system.validate_config_overrides(
                messages          = self.config.messages,
                overrides         = config_overrides,
                wandb_integration = self.config.wandb_integration
            )

        if issues:
            self.ui.print_message(self.config.messages.validation["config_issues"], "warning")
            for issue in issues:
                self.ui.console.print(f"  [warning]⚠️  {issue}[/warning]")
        else:
            self.ui.print_message(
                message  = self.config.messages.validation["config_passed"],
                msg_type = "success"
            )

        self.ui.print_major_section("Integration Check")
        status, details = self.system.check_wandb_status(self.config)

        if "Not" in status:
            self.ui.print_message(f"wandb: {details}", "warning")
        else:
            self.ui.print_message(f"wandb: {details}", "success")

        self.ui.console.print()
        if issues or "Not" in status:
            self.ui.print_message(
                message  = self.config.messages.validation["with_warnings"],
                msg_type = "warning"
            )
            self.ui.print_message(
                message  = self.config.messages.validation["review_issues"],
                msg_type = "tip"
            )
        else:
            self.ui.print_message(
                message  = self.config.messages.validation["all_passed"],
                msg_type = "success"
            )
            self.ui.print_message(
                message  = self.config.messages.validation["system_ready"],
                msg_type = "success"
            )