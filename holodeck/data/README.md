# Holodeck Data Directory

This directory manages the data assets and external binary files required by the `holodeck` simulation suite. 

To keep the repository lightweight and ensure rapid cloning, large binary datasets (such as `.hdf5`, `.tar.gz`, and `.pkl` files) are hosted externally on Zenodo and downloaded automatically on-demand using `pooch`.

## For Users

You do not need to manually download or place data files into this directory. When you initialize a module or call a population configuration that requires external data (e.g., `population.Pop_Illustris()`), `holodeck` will seamlessly fetch the required files from our production data repository, verify their integrity via cryptographic hashes, and cache them locally.

Local cache paths are managed dynamically. If a file is missing or corrupted, the package will automatically attempt to redownload it.

## For Developers

### Adding a New Large Dataset

If your development work or a new physics module introduces a large static data file (typically anything over a few hundred kilobytes, or any binary format), **do not commit the raw file to Git history**. Instead, follow this workflow:

1. **Upload the file to Zenodo**: 
   Add the asset to the project's production Zenodo deposition (Record ID: `20534588`) or, better yet, to your own Zenodo [or other public server] deposition..
2. **Generate the SHA256 Hash**:
   Calculate the cryptographic hash of your file. You can do this in Python using Pooch:
   ```python
   import pooch
   print(pooch.file_hash("path/to/your/file.hdf5"))
   ```
3. **Update `registry.txt`**:
   Add a line to `holodeck/data/registry.txt` matching the three-column format (do not include `holodeck/data/` in your file path):
   ```
   path/to/filename.ext <SHA256_HASH> https://zenodo.org/records/20534588/files/filename.ext
   ``` 
_Note: Zenodo stores uploaded files in a flat structure, so the third column URL should use the base filename, while the first column retains the internal directory structure you want Pooch to recreate locally._   
4. **Use `get_data_path` in your code**:
Instead of hardcoding a local file path, always resolve the path dynamically using the data managre:
```python
from holodeck.data_manager import get_data_path
data_file_path = get_data_path("path/to/filename.ext")
```

### File Tracking Rules (`.gitignore`)
The `.gitignore` configuration for this directory is set up to block large binary additions while preserving critical configuration, setup, and documentation tracking:
- `registry.txt` and this `README.md` must always be tracked by Git.
- Core text/asset files under Git control can be explicitly whitelisted if they require or would benefit from direct version history tracking as with code edits.  But consider uploading them to Zenodo as well so that the data files may be discovered together.

## Policy on Binary and Large Files

To keep the `holodeck` codebase lightweight, high-performance, and fast to clone, **no large files (>5 MB) or binary formats (`.hdf5`, `.npz`, `.pkl`, `.pdf`, etc.) should be committed directly to the repository history.** 

### Why this rule exists
Committing scientific datasets, uncompressed text simulation tables, or reference PDFs permanently inflates Git history packfiles. Even if a heavy file is deleted or replaced in a subsequent commit, it remains baked into the hidden `.git/objects/` database forever. This forces every developer down the road to download massive, dead historical artifacts during a routine `git clone`.


