# ============================================================================
# ▶  USER CONFIGURATION — edit these before running
# ============================================================================

RAW_DATA_DIR = r"C:\Users\Lenovo\Projects\Project1_CVAT_YOLO\data\obj_data"
OUTPUT_DIR = r"C:\Users\Lenovo\Projects\Project1_CVAT_YOLO\data\dev_pipeline_test"

# Classes to detect (must match CVAT labels)
CLASS_NAMES = ["airplane", "boat", "ship", "car"]

# Data split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ============================================================================
# ▶  IMPORTS
# ============================================================================

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import random
import shutil


# ============================================================================
# ▶  HELPER FUNCTIONS
# ============================================================================

# Get all image files in the raw data directory

def get_image_files(directory):
    """Get list of image files in a directory."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

# Match labels to images

def get_label_file(image_file):
    """Get corresponding label file for an image."""
    base_name = os.path.splitext(image_file)[0]
    return f"{base_name}.txt"

# Split data into train/val/test sets (shuffle before splitting)

def split_data(image_files, train_ratio, val_ratio, test_ratio):
    """Split data into train, val, and test sets."""
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return (image_files[:train_end], 
            image_files[train_end:val_end], 
            image_files[val_end:])

# Create output directories

def create_output_dirs(base_dir):
    """Create output directories for train, val, and test sets."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# Copy files to output directories

def copy_files(image_files, split, output_dir):
    """Copy image and label files to the appropriate output directories."""
    for image_file in image_files:
        label_file = get_label_file(image_file)
        shutil.copy(os.path.join(RAW_DATA_DIR, image_file), 
                    os.path.join(output_dir, split, 'images', image_file))
        shutil.copy(os.path.join(RAW_DATA_DIR, label_file), 
                    os.path.join(output_dir, split, 'labels', label_file))
        
# Create dataset.yaml file

def create_dataset_yaml(output_dir, class_names):
    yaml_content = f"""path: {output_dir}
    train: train/images
    val: val/images
    test: test/images

    names:
    """
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)


# ============================================================================
# ▶  VISUALIZATION 
# ============================================================================

def visualize_class_distribution(image_files):
    """Visualize class distribution in the dataset."""
    class_counts = Counter()
    
    for image_file in image_files:
        label_file = get_label_file(image_file)
        with open(os.path.join(RAW_DATA_DIR, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    # Map class IDs to class names
    classes = [CLASS_NAMES[class_id] for class_id in sorted(class_counts.keys())]
    counts = [class_counts[class_id] for class_id in sorted(class_counts.keys())]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate class colors from CLASS_NAMES
    cmap = plt.get_cmap("Set2")
    color_map = {
        cls: cmap(i % cmap.N)
        for i, cls in enumerate(CLASS_NAMES)
    }
    
    # Validate detected classes
    unknown_classes = [cls for cls in classes if cls not in color_map]
    if unknown_classes:
        raise ValueError(f"Unexpected classes detected: {unknown_classes}")
    
    # Assign colors to bars based on class
    bar_colors = [color_map[cls] for cls in classes]
    
    bars = ax.bar(classes, counts, color=bar_colors, edgecolor="black", linewidth=1, alpha=0.8, width=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=13)
    
    ax.set_xlabel("Object Class", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Detected Objects by Class\nTotal: {sum(counts)}", 
                 fontsize=15, fontweight="bold")
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    
    # Ensure y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Ensure directory exists before saving
    save_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {save_path}")


# ============================================================================
# ▶  MAIN PIPELINE
# ============================================================================

def main():
    # Step 1: Get all image files
    image_files = get_image_files(RAW_DATA_DIR)
    
    # Step 2: Split data into train/val/test sets
    train_files, val_files, test_files = split_data(image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Step 3: Create output directories
    create_output_dirs(OUTPUT_DIR)
    
    # Step 4: Copy files to output directories
    copy_files(train_files, 'train', OUTPUT_DIR)
    copy_files(val_files, 'val', OUTPUT_DIR)
    copy_files(test_files, 'test', OUTPUT_DIR)
    
    # Step 5: Create dataset.yaml file
    create_dataset_yaml(OUTPUT_DIR, CLASS_NAMES)

    # Step 6: Visualize class distribution (optional)
    visualize_class_distribution(image_files)

    print("Data pipeline completed successfully!")

if __name__ == "__main__":
    main()  
