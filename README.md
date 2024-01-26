[![PyPI version](https://badge.fury.io/py/embedman.svg)](https://badge.fury.io/py/embedman)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/embedman)](https://pepy.tech/project/embedman)

# EmbedMan

`EmbedMan` is a Python package designed to manage embeddings for code analysis efficiently. It facilitates the process of generating and retrieving embeddings from a specified directory of code files, utilizing the power of language models and embedding storage solutions.

## Installation

To install `EmbedMan`, you can use pip:

```bash
pip install embedman
```

## Usage

### As a Python Module

After installation, `EmbedMan` can be imported and used in your Python projects.

Example:

```python
from embed_man import EmbedManager

# Initialize the EmbedManager with desired parameters
embed_manager = EmbedManager(
    path="path/to/your/code/directory",
    glob_rule="**/*.py",
    use_cache=True
)

# Run the embedding process and get a retriever for querying embeddings
retriever = embed_manager.run()

# You can now use the retriever to query embeddings
```

### Configurable Parameters

`EmbedMan` allows various configurations to tailor the embedding process to your needs, including:

- `path`: The directory path to scan for documents.
- `glob_rule`: Glob pattern to match files within the directory.
- `suffixes`: List of file suffixes to include.
- `exclude`: List of patterns to exclude.
- `language`: Programming language of the documents.
- `parser_threshold`: Threshold for the parser to consider a document valid.
- `chunk_size`: Size of chunks to split documents into for embedding.
- `chunk_overlap`: Overlap between consecutive chunks.
- `cache_dir`: Directory path for caching embeddings.
- `namespace_cache`: Namespace for the cache to avoid collisions.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/chigwell/embedman/issues).

## License

[MIT](https://choosealicense.com/licenses/mit/)
