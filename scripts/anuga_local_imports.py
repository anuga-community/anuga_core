#!/usr/bin/env python3

# List local imports by scanning and filtering a source file
import ast
import os

def find_local_imports(filepath, project_root):
    with open(filepath) as f:
        tree = ast.parse(f.read())

    #print(f"Scanning {filepath} for local imports...")
    #print(f"Using project root: {project_root}")
    #print(tree)

    imports = []
    for node in ast.walk(tree):
        #print(f"Visiting node: {node}")
        if isinstance(node, ast.Import):
            for n in node.names:
                module_name = n.name.split('.')[0]
                #print(f"Found import: {module_name}")
                if os.path.isfile(os.path.join(project_root, module_name + '.py')):
                    imports.append(module_name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split('.')[0] if node.module else None
            #print(f"Found from-import: {module_name}")
            if module_name and os.path.isfile(os.path.join(project_root, module_name + '.py')):
                imports.append(module_name)
    
    imports = list(set(imports))  # Remove duplicates

    return list({name + ".py" for name in imports})
    return imports

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Find local imports."
    )

    parser.add_argument(
        "script_file",
        type=str,
        nargs="?",
        default="run_small_towradgi.py",
        help="The Python script test (default: run_small_towradgi.py)"
    )

    parser.add_argument(
        "project_root",
        type=str, 
        default=" ",
        nargs=argparse.REMAINDER,
        help="Project root directory (default: current directory)")

    args = parser.parse_args()
    script_file = args.script_file
    project_root = args.project_root

    #print(f"Using script file: {script_file}")
    #print(f"Using project root: {project_root}")

    if project_root == []:
        project_root = os.path.dirname(os.path.abspath(script_file))
    else:
        project_root = os.path.dirname(os.path.abspath(project_root[0]))

    #print(f"Resolved project root: {project_root}")

    imports = find_local_imports(script_file, project_root)
    print("Local imports found:", imports)