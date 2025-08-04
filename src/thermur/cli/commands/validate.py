"""
Validation command for the Thermur CLI.

Allows users to check their system setup and configuration syntax
without initiating a full training run.
"""
from thermur.cli import app
from typer       import Option


def validate(
    overrides : list[str] | None = Option(
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
    app.ui.print_header("System Validation")
    app.ui.display_system_validation(app.system)

    app.ui.print_section("Configuration Check")

    with app.ui.console.status(
        "[accent]Validating configuration...[/accent]",
        spinner = "dots"
    ):
        issues = app.system.validate_overrides(overrides)

    app.ui.print_message(
        message  = (
            "Configuration issues found:" if issues
            else "Configuration validation passed!"
        ),
        msg_type = "warning" if issues else "success"
    )

    for i, issue in enumerate(issues, start=1):
        app.ui.console.print(f"  [warning]⚠️  {i}. {issue}[/warning]")

    app.ui.console.print()

    if issues:
        app.ui.print_message(
            message  = "Validation completed with warnings",
            msg_type = "warning"
        )
        app.ui.print_message(
            message  = "Review the issues above before training",
            msg_type = "tip"
        )
    else:
        app.ui.print_message(
            message  = "All validations passed!",
            msg_type = "success"
        )
        app.ui.print_message(
            message  = "Your system is ready for training",
            msg_type = "success"
        )
