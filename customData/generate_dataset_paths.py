import os
import argparse

# Function to get all file paths in a directory
def get_file_paths(base_path):
    file_paths = []
    classes = os.listdir(base_path)
    for class_dir in classes:
        full_path = os.path.join(base_path, class_dir)
        if os.path.isdir(full_path):
            files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.JPEG')]
            file_paths.extend(files)
    return file_paths

def main(base_dataset_path):
    # Define the paths for the train and test datasets
    train_base_path = os.path.join(base_dataset_path, 'train')
    test_base_path = os.path.join(base_dataset_path, 'val')

    # Get file paths for train and test datasets
    train_files = get_file_paths(train_base_path)
    test_files = get_file_paths(test_base_path)

    # Write file paths to train.txt and test.txt
    with open('train.txt', 'w') as file:
        file.write('\n'.join(train_files))

    with open('test.txt', 'w') as file:
        file.write('\n'.join(test_files))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate file paths for train and test datasets.")
    parser.add_argument("base_dataset_path", type=str, help="Base path for the dataset")
    
    args = parser.parse_args()
    main(args.base_dataset_path)
