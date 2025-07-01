#!/usr/bin/env python3
"""Debug the exact issue with the table rendering."""

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.cells import cell_len

console = Console()

# First, let's see what's actually happening with cell widths
print("=== Cell Width Analysis ===")
features = [
    "🔥 Thermal Constraints",
    "🪽 Flock Coordination",
    "🧠 Imitation Learning",
    "🪄 wandb Integration",
    "🎯 Hydra Configuration",
    "📈 Live Visualization"
]

for f in features:
    cells = cell_len(f)
    chars = len(f)
    print(f"{f:<30} chars={chars:2d} cells={cells:2d}")

# Now let's try different table rendering approaches
print("\n=== Approach 1: Standard Table ===")
table1 = Table(box=box.MINIMAL, show_edge=False, padding=(0, 1))
table1.add_column("Feature", width=28, style="bright_cyan")
table1.add_column("Description", width=60)

descriptions = [
    "Physics-based heat modeling for drone safety",
    "Multi-agent flocking with obstacle avoidance",
    "Expert policy cloning with BC and DAgger",
    "Real-time experiment tracking and visualization",
    "Modular config system with validation",
    "3D rendering of flock dynamics"
]

for i, feature in enumerate(features):
    table1.add_row(feature, descriptions[i])
console.print(table1)

# Approach 2: Force column width with overflow
print("\n=== Approach 2: Fixed Width with Overflow ===")
table2 = Table(box=box.MINIMAL, show_edge=False, padding=(0, 1))
table2.add_column("Feature", width=28, style="bright_cyan", overflow="fold")
table2.add_column("Description", width=60)

for i, feature in enumerate(features):
    table2.add_row(feature, descriptions[i])
console.print(table2)

# Approach 3: Using Text objects with explicit width
print("\n=== Approach 3: Text with set_length ===")
table3 = Table(box=box.MINIMAL, show_edge=False, padding=(0, 1))
table3.add_column("Feature", width=28, style="bright_cyan")
table3.add_column("Description", width=60)

for i, feature in enumerate(features):
    text = Text(feature)
    # Force text to be exactly 28 cells wide
    text = text.truncate(28, overflow="fold", pad=True)
    table3.add_row(text, descriptions[i])
console.print(table3)

# Approach 4: Manual padding based on actual cell width
print("\n=== Approach 4: Manual Cell Width Compensation ===")
table4 = Table(box=box.MINIMAL, show_edge=False, padding=(0, 1))
table4.add_column("Feature", width=28, style="bright_cyan")
table4.add_column("Description", width=60)

double_width_emojis = {'🪽', '🪄'}
for i, feature in enumerate(features):
    # Calculate how many cells the text actually takes
    actual_cells = cell_len(feature)
    
    # If it contains double-width emoji and cells are less than expected
    if any(e in feature for e in double_width_emojis):
        # Rich might be miscalculating, so add padding
        feature_text = Text(feature + " " * (28 - actual_cells + 1))
    else:
        feature_text = Text(feature)
    
    table4.add_row(feature_text, descriptions[i])
console.print(table4)

# Let's also check if it's a font/terminal issue
print("\n=== Terminal Rendering Test ===")
print("Column markers at position 28:")
print("1234567890123456789012345678|")
for f in features:
    print(f"{f:<28}|")

print("\nWith manual spacing adjustment:")
print("1234567890123456789012345678|")
for f in features:
    if any(e in f for e in double_width_emojis):
        print(f"{f:<27} |")  # One less padding
    else:
        print(f"{f:<28}|")