from datetime import datetime, timedelta
from pathlib import Path
import shutil
import os
import sys
if hasattr(sys, 'ps1'):
    print("💾 Running in interactive mode. Be sure all files are saved.")
else:
    print("🔍 Ensuring all files are saved before proceeding.")

# === CONFIGURATION ===
N_DAYS = 3
BACKUP_ROOT = Path("word_backups")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
backup_dir = BACKUP_ROOT / timestamp

# === STEP 1: Create timestamped backup subfolder ===
backup_dir.mkdir(parents=True, exist_ok=True)

# === STEP 2: Copy templates/*.html ===
template_dir = Path("templates")
for html_file in template_dir.glob("*.html"):
    dst_file = backup_dir / f"{html_file.stem}.txt"
    shutil.copy2(html_file, dst_file)
    print(f"✅ Copied: {html_file} → {dst_file}")

# === STEP 3: Copy top-level files ===
top_level_files = [
    ".env", ".env.local", ".gitignore", "backup_script.py", "forms.py",
    "main.py", "migrate.py", "Procfile", "requirements.txt", "wsgi.py",
]

for filename in top_level_files:
    src = Path(filename)
    if src.exists():
        dst_file = backup_dir / f"{src.stem}.txt"
        shutil.copy2(src, dst_file)
        print(f"✅ Copied: {src} → {dst_file}")
    else:
        print(f"⚠️ Skipped (not found): {src}")

# === STEP 4: Delete folders older than N_DAYS ===
cutoff = datetime.now() - timedelta(days=N_DAYS)

for folder in BACKUP_ROOT.iterdir():
    if folder.is_dir():
        try:
            folder_time = datetime.strptime(folder.name, "%Y-%m-%d_%H-%M")
            if folder_time < cutoff:
                shutil.rmtree(folder)
                print(f"🗑️ Deleted old backup: {folder}")
        except ValueError:
            print(f"⚠️ Skipped (invalid format): {folder.name}")
