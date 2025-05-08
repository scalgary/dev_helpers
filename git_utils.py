import subprocess
from collections import defaultdict
from pathlib import Path

def get_git_tracked_tree(repo_path="."):
    """Affiche une structure de type 'tree' avec les fichiers trackés par Git."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    files = result.stdout.strip().split('\n')
    
    # Construire un arbre hiérarchique
    tree = lambda: defaultdict(tree)
    root = tree()
    
    for file_path in files:
        parts = file_path.split('/')
        current = root
        for part in parts:
            current = current[part]
    
    def print_tree(node, prefix=""):
        for i, key in enumerate(sorted(node)):
            connector = "└── " if i == len(node) - 1 else "├── "
            print(prefix + connector + key)
            if node[key]:
                extension = "    " if i == len(node) - 1 else "│   "
                print_tree(node[key], prefix + extension)

    print(f"{repo_path.rstrip('/')}/")
    print_tree(root)

if __name__ == "__main__":
    get_git_tracked_tree()