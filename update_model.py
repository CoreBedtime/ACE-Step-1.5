import os
import shutil
from pathlib import Path

def update_base_model():
    """
    Overwrites the base model implementation in the checkpoint folder 
    with the modified versions from the project root.
    """
    root_dir = Path(__file__).parent
    checkpoint_dir = root_dir / "checkpoints" / "acestep-v15-base"
    
    # List of files to update
    files_to_update = [
        "modeling_acestep_v15_base.py",
        "configuration_acestep_v15.py",
        "apg_guidance.py"
    ]
    
    if not checkpoint_dir.exists():
        print(f"❌ Error: Checkpoint directory not found at {checkpoint_dir}")
        return

    print(f"🚀 Updating base model implementation in {checkpoint_dir}...")
    
    updated_count = 0
    for filename in files_to_update:
        src_path = root_dir / filename
        dst_path = checkpoint_dir / filename
        
        if src_path.exists():
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  ✅ Updated: {filename}")
                updated_count += 1
            except Exception as e:
                print(f"  ❌ Failed to copy {filename}: {e}")
        else:
            print(f"  ⚠️ Skip: {filename} not found in root directory")

    if updated_count == len(files_to_update):
        print(f"\n✨ Successfully updated all {updated_count} implementation files.")
    else:
        print(f"\n⚠️ Update finished with some skips/errors ({updated_count}/{len(files_to_update)} files updated).")

if __name__ == "__main__":
    update_base_model()