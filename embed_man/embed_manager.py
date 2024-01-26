"""
    embed_man/embed_manager.py:
    Manages the generation and retrieval of embeddings for code analysis.
"""
import os
import logging
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


class EmbedManager:
    """Manages the generation and retrieval of embeddings for code analysis.

    This class provides a streamlined process for loading documents from a specified
    directory, splitting them into manageable chunks, generating embeddings for these chunks,
    and creating a retriever for querying these embeddings.

    Attributes:
        path (str): Directory path to scan for documents.
        glob_rule (str): Glob pattern to match files within the directory.
        suffixes (list): List of file suffixes to include.
        exclude (list): List of patterns to exclude.
        language (Language): Programming language of the documents.
        parser_threshold (int): Threshold for the parser to consider a document valid.
        logging_level (Optional[str]): Logging level to use for output messages.
        chunk_size (int): Size of chunks to split documents into for embedding.
        chunk_overlap (int): Overlap between consecutive chunks.
        cache_dir (str): Directory path for caching embeddings.
        namespace_cache (str): Namespace for the cache to avoid collisions.
        search_type (str): Type of search to perform with the embeddings.
        search_kwargs (dict): Additional keyword arguments for the search.
        use_cache (bool): Flag to determine if caching should be used.
    """

    def __init__(self, path=os.getcwd(), glob_rule="**/*", suffixes=[".py"],
                 exclude=["**/non-utf8-encoding.py"], language=Language.PYTHON,
                 parser_threshold=500, logging_level=None, chunk_size=2000,
                 chunk_overlap=200, cache_dir=os.path.join(os.getcwd(), "cache"),
                 namespace_cache="cache_embeddings", search_type="mmr",
                 search_kwargs={"k": 8}, use_cache=True):
        """Initializes the EmbedManager with configurable parameters."""

        self.path = path
        self.glob_rule = glob_rule
        self.suffixes = suffixes
        self.exclude = exclude
        self.language = language
        self.parser_threshold = parser_threshold
        self.logging_level = logging_level
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir
        self.namespace_cache = namespace_cache
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.use_cache = use_cache

        # Set up logging
        if logging_level:
            logging.basicConfig(level=logging_level)

    def load_documents(self):
        """Loads documents from the filesystem based on the initialized configuration.

        Returns:
            list: A list of loaded documents.
        """

        loader = GenericLoader.from_filesystem(
            self.path,
            glob=self.glob_rule,
            suffixes=self.suffixes,
            exclude=self.exclude,
            parser=LanguageParser(language=self.language, parser_threshold=self.parser_threshold),
        )
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents):
        """Splits documents into chunks suitable for embedding.

        Args:
            documents (list): A list of documents to be split.

        Returns:
            list: A list of text chunks ready for embedding.
        """

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=self.language, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        texts = splitter.split_documents(documents)
        logging.info(f"Split into {len(texts)} text chunks")
        return texts

    def generate_embeddings(self, texts, use_cache=True):
        """Generates embeddings for the provided text chunks, optionally using a cache.

        Args:
            texts (list): A list of text chunks for which to generate embeddings.
            use_cache (bool): Whether to use cached embeddings if available.

        Returns:
            Retriever: A retriever instance for querying the generated embeddings.
        """
        if use_cache:
            cache_store = LocalFileStore(self.cache_dir)
            embedder = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings=GPT4AllEmbeddings(),
                document_embedding_cache=cache_store,
                namespace=self.namespace_cache
            )
        else:
            embedder = GPT4AllEmbeddings()

        db = Chroma.from_documents(texts, embedder)
        return db.as_retriever(search_type=self.search_type, search_kwargs=self.search_kwargs)

    def run(self):
        """Executes the main process of loading, splitting, embedding, and retrieving documents.

        This method orchestrates the entire process from document loading to creating
        a retriever for the generated embeddings.

        Returns:
            Retriever: A retriever instance for querying the generated embeddings.

        Raises:
            Exception: Propagates any exception that occurs during the process.
        """

        try:
            documents = self.load_documents()
            texts = self.split_documents(documents)
            retriever = self.generate_embeddings(texts, use_cache=self.use_cache)
            return retriever
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise
