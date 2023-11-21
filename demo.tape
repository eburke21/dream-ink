# VHS Demo: DreamInk CLI
# https://github.com/charmbracelet/vhs

Output demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1000
Set Height 600
Set Padding 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 0.5

Env PATH "/Users/erinburke/Library/Caches/pypoetry/virtualenvs/dreamink-eEGVeVP9-py3.12/bin:/opt/homebrew/bin:/usr/bin:/bin"

# --- Main help ---
Type "dreamink --help"
Enter
Sleep 3s
Ctrl+L

# --- Add: record a dream entry ---
Type "dreamink add --help"
Enter
Sleep 3s
Ctrl+L

# --- Generate: illustrate entries with DALL-E 3 ---
Type "dreamink generate --help"
Enter
Sleep 3s
Ctrl+L

# --- Compare: same dream across multiple art styles ---
Type "dreamink compare --help"
Enter
Sleep 3s
Ctrl+L

# --- Tags: motif frequency and co-occurrence ---
Type "dreamink tags --help"
Enter
Sleep 3s
Ctrl+L

# --- Journal + Export ---
Type "dreamink journal --help"
Enter
Sleep 2.5s
Ctrl+L

Type "dreamink export --help"
Enter
Sleep 2.5s
Ctrl+L

# --- Test suite: 119 tests, fully mocked ---
Type "pytest tests/ -v"
Enter
Sleep 10s

# Let the final results sit
Sleep 3s
