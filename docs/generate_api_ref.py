#!/usr/bin/env python3
from astropy.coordinates import matrix_utilities
import os
import sys

# Define directories relative to this script
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(DOCS_DIR, "source", "api_ref")
REPO_ROOT = os.path.dirname(DOCS_DIR)
CODE_DIR = os.path.join(REPO_ROOT, "holodeck")

# 1. Global Ignore Configuration
IGNORE_DIRS = {"__pycache__", "tests", "data", "source", "build"}
IGNORE_FILES = {
    "__init__.py",
    "param_spaces_old.py",
    "relations.py",
}

# 2. Categorization Mapping for holodeck.rst
CATEGORIES = {
    "Core Package Foundations": [
        "holodeck.constants",
        "holodeck.cyutils",
        "holodeck.discrete_cyutils",
        "holodeck.extensions",
        "holodeck.logger",
    ],
    "Astrophysics & Population Modeling": [
        "holodeck.accretion",
        "holodeck.discrete",
        "holodeck.galaxy_profiles",
        "holodeck.host_relations",
        "holodeck.pop_observational",
        "holodeck.sams",
    ],
    "Gravitational Waves & Evolution": [
        "holodeck.anisotropy",
        "holodeck.gravwaves",
        "holodeck.hardening",
        "holodeck.single_sources",
    ],
    "Analysis, Libraries & Tools": [
        "holodeck.detstats",
        "holodeck.gps",
        "holodeck.librarian",
        "holodeck.observations",
        "holodeck.param_spaces",
        "holodeck.plot",
        "holodeck.utils",
    ],
}

def clean_and_prepare():
    """Ensure a fresh, clean api_ref directory target."""
    print(f"Cleaning target directory: {SOURCE_DIR}")
    if os.path.exists(SOURCE_DIR):
        import shutil
        shutil.rmtree(SOURCE_DIR)
    os.makedirs(SOURCE_DIR, exist_ok=True)

def write_rst_file(module_name, is_package=False, submodules=None, pure_structure=False):
    """Generate individual .rst files for modules and packages cleanly."""
    file_path = os.path.join(SOURCE_DIR, f"{module_name}.rst")
    
    title = f"{module_name} package" if is_package else f"{module_name} module"
    underline = "=" * len(title)
    
    content = [
        title,
        underline,
        ""
    ]
    
    # Only try to extract docstrings if it's an importable Python file/package
    if not pure_structure:
        content.extend([
            f".. automodule:: {module_name}",
            "   :members:",
            "   :undoc-members:",
            "   :show-inheritance:",
            "   :member-order: bysource",
            ""
        ])
    
    # If it's a directory package containing sub-items, link them together
    if is_package and submodules:
        content.extend([
            "Submodules",
            "----------",
            "",
            ".. toctree::",
            "   :maxdepth: 4",
            ""
        ])
        for sub in sorted(submodules):
            content.append(f"   {sub}")
        content.append("")
        
    with open(file_path, "w") as f:
        f.write("\n".join(content))


def scan_codebase():
    """Traverse holodeck codebase and construct the .rst files dynamically."""
    print(f"Scanning codebase: {CODE_DIR}")
    
    # Track items discovered to link in top-level maps
    top_level_modules = set()
    
    for root, dirs, files in os.walk(CODE_DIR):
        # Apply directory filters in-place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        # Determine package identity path
        rel_path = os.path.relpath(root, REPO_ROOT)
        package_prefix = rel_path.replace(os.sep, ".")
        
# If we are inside a subpackage directory (like holodeck/sams)
        if root != CODE_DIR:
            # ONLY treat this directory as an importable package if it contains an __init__.py
            has_init = "__init__.py" in files
            
            submodules = []
            for f in files:
                if (f.endswith(".py") or f.endswith(".pyx")) and f not in IGNORE_FILES:
                    mod_base = os.path.splitext(f)[0]
                    full_mod_name = f"{package_prefix}.{mod_base}"
                    write_rst_file(full_mod_name, is_package=False)
                    submodules.append(full_mod_name)
            
            if submodules and has_init:
                write_rst_file(package_prefix, is_package=True, submodules=submodules)
            elif submodules and not has_init:
                # If there's no __init__.py, don't use .. automodule:: on the directory path itself
                write_rst_file(package_prefix, is_package=True, submodules=submodules, pure_structure=True)
            continue

        # Processing the top-level holodeck/ root folder items
        for f in files:
            if (f.endswith(".py") or f.endswith(".pyx")) and f not in IGNORE_FILES:
                mod_base = os.path.splitext(f)[0]
                full_mod_name = f"holodeck.{mod_base}"
                write_rst_file(full_mod_name, is_package=False)
                top_level_modules.add(full_mod_name)
                
        for d in dirs:
            # Mark top level tracking directory targets (like holodeck.sams)
            top_level_modules.add(f"holodeck.{d}")

def generate_master_index():
    """Regenerate the unified holodeck.rst entry point layout."""
    master_path = os.path.join(SOURCE_DIR, "holodeck.rst")
    print(f"Generating master index layout: {master_path}")
    
    content = [
        "======================",
        "holodeck API Reference",
        "======================",
        "",
        ".. automodule:: holodeck",
        "    :members:",
        "    :private-members:",
        "    :undoc-members:",
        "    :show-inheritance:",
        "    :member-order: bysource",
        ""
    ]
    
    # Iterate clean structural layout blocks
    for category_name, modules in CATEGORIES.items():
        content.extend([
            ".. toctree::",
            "   :maxdepth: 2",
            f"   :caption: {category_name}",
            ""
        ])
        for mod in sorted(modules):
            content.append(f"   {mod}")
        content.extend(["", ""])
        
    with open(master_path, "w") as f:
        f.write("\n".join(content))

if __name__ == "__main__":
    clean_and_prepare()
    scan_codebase()
    generate_master_index()
    print("✨ API Reference Generation Complete!")
