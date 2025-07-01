#!/usr/bin/env python3
"""Iterative test to find correct emoji spacing for Rich tables."""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# Test configurations with different spacing approaches
double_width_emojis = {'🪽', '🪄'}

def create_test_table(test_name, spacing_strategy):
    """Create a test table with given spacing strategy."""
    console.print(f"\n[bold yellow]{test_name}[/bold yellow]")
    
    table = Table(
        header_style="bold bright_cyan",
        border_style="bright_blue",
        box=box.MINIMAL,
        show_edge=False,
        padding=(0, 1),
        expand=False
    )
    
    table.add_column("Feature", style="bright_cyan", width=28)
    table.add_column("Description", style="white", width=60)
    
    features = [
        ("🔥 Thermal Constraints", "Physics-based heat modeling for drone safety"),
        ("🪽 Flock Coordination", "Multi-agent flocking with obstacle avoidance"),
        ("🧠 Imitation Learning", "Expert policy cloning with BC and DAgger"),
        ("🪄 wandb Integration", "Real-time experiment tracking and visualization"),
        ("🎯 Hydra Configuration", "Modular config system with validation"),
        ("📈 Live Visualization", "3D rendering of flock dynamics")
    ]
    
    for name, desc in features:
        # Apply spacing strategy
        modified_name = spacing_strategy(name, double_width_emojis)
        table.add_row(Text(modified_name, no_wrap=True), Text(desc, no_wrap=True))
    
    console.print(table)
    
    # Also print raw text to see alignment
    console.print("\n[dim]Raw text alignment check:[/dim]")
    for name, _ in features:
        modified_name = spacing_strategy(name, double_width_emojis)
        console.print(f"{modified_name:<28}|")

# Different spacing strategies to test
strategies = [
    ("No modification", 
     lambda name, dw: name),
    
    ("Add 1 space at end for double-width", 
     lambda name, dw: name + " " if any(e in name for e in dw) else name),
    
    ("Add 2 spaces at end for double-width", 
     lambda name, dw: name + "  " if any(e in name for e in dw) else name),
    
    ("Remove 1 space from middle for double-width",
     lambda name, dw: name.replace(" ", "", 1) if any(e in name for e in dw) else name),
    
    ("Add space before text for double-width",
     lambda name, dw: name.replace(" ", "  ", 1) if any(e in name for e in dw) else name),
    
    ("Use non-breaking space",
     lambda name, dw: name + "\u00A0" if any(e in name for e in dw) else name),
    
    ("Use zero-width space", 
     lambda name, dw: name + "\u200B" if any(e in name for e in dw) else name),
    
    ("Use thin space",
     lambda name, dw: name + "\u2009" if any(e in name for e in dw) else name),
]

# Run all tests
for strategy_name, strategy_func in strategies:
    create_test_table(strategy_name, strategy_func)

# Final test - let's check exact character positions
console.print("\n[bold cyan]Character position analysis:[/bold cyan]")
test_strings = [
    "🔥 Thermal Constraints",
    "🪽 Flock Coordination",
    "🧠 Imitation Learning",
    "🪄 wandb Integration",
]

for s in test_strings:
    console.print(f"\n'{s}':")
    console.print("Position: " + "".join(str(i % 10) for i in range(len(s) + 5)))
    console.print("String:   " + s + "|")
    from rich.cells import cell_len
    console.print(f"Length: {len(s)} chars, {cell_len(s)} cells")