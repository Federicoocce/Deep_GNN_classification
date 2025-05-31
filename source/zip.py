import tarfile
import os
from pathlib import Path


def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.

    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")


if __name__ == "__main__":
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    submission_dir = os.path.join(base_dir, 'submission')  # Path to the folder you want to compress
    output_file = os.path.join(base_dir, 'submission.gz')  # Output .gz file name
    gzip_folder(submission_dir, output_file)