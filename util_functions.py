# -*- coding: utf-8 -*-

import subprocess
import os

def clone_folder_from_git(repo_url, folder_path, local_dir):
    """
    Clone a specific folder from a Git repository using sparse checkout.

    Parameters:
    - repo_url: str, the git repository URL.
    - folder_path: str, the folder inside the repo to clone (relative path).
    - local_dir: str, the local directory to clone into.
    """

    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists. Remove or choose another path.")
        return

    try:
        # Initialize a new git repo
        subprocess.run(['git', 'init', local_dir], check=True)

        # Set repo URL as origin
        subprocess.run(['git', '-C', local_dir, 'remote', 'add', '-f', 'origin', repo_url], check=True)

        # Enable sparse checkout
        subprocess.run(['git', '-C', local_dir, 'config', 'core.sparseCheckout', 'true'], check=True)

        # Write the folder path to sparse-checkout file
        sparse_checkout_file = os.path.join(local_dir, '.git', 'info', 'sparse-checkout')
        with open(sparse_checkout_file, 'w') as f:
            f.write(folder_path + '/\n')

        # Pull only the specified folder
        subprocess.run(['git', '-C', local_dir, 'pull', 'origin', 'main'], check=True)  # or 'master' branch

        print(f"Successfully cloned folder '{folder_path}' from {repo_url} into {local_dir}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Example usage:
# clone_folder_from_git("https://github.com/user/repo.git", "path/to/folder", "./local_folder")

