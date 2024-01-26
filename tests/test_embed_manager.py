import unittest
from embed_man.embed_manager import EmbedManager
from unittest.mock import patch, MagicMock

class TestEmbedManager(unittest.TestCase):
    def setUp(self):
        # Setup for test cases
        self.embed_manager = EmbedManager(path="test_path", use_cache=False)

    @patch("embed_man.embed_manager.GenericLoader")
    def test_load_documents(self, mock_loader):
        # Mock the loader to return a specific value
        mock_loader.from_filesystem.return_value.load.return_value = ["doc1", "doc2"]
        documents = self.embed_manager.load_documents()
        self.assertEqual(len(documents), 2)

    @patch("embed_man.embed_manager.RecursiveCharacterTextSplitter")
    def test_split_documents(self, mock_splitter):
        # Mock the splitter to return specific chunks
        mock_splitter.from_language.return_value.split_documents.return_value = ["chunk1", "chunk2"]
        chunks = self.embed_manager.split_documents(["doc1", "doc2"])
        self.assertEqual(len(chunks), 2)

    @patch("embed_man.embed_manager.Chroma")
    @patch("embed_man.embed_manager.CacheBackedEmbeddings")
    @patch("embed_man.embed_manager.LocalFileStore")
    def test_generate_embeddings(self, mock_store, mock_cache_embeddings, mock_chroma):
        # Mock the embeddings and Chroma initialization
        mock_chroma.from_documents.return_value.as_retriever.return_value = MagicMock()
        retriever = self.embed_manager.generate_embeddings(["chunk1", "chunk2"])
        self.assertTrue(retriever)

    @patch("embed_man.embed_manager.EmbedManager.load_documents")
    @patch("embed_man.embed_manager.EmbedManager.split_documents")
    @patch("embed_man.embed_manager.EmbedManager.generate_embeddings")
    def test_run(self, mock_generate_embeddings, mock_split_documents, mock_load_documents):
        # Mock the entire process
        mock_load_documents.return_value = ["doc1", "doc2"]
        mock_split_documents.return_value = ["chunk1", "chunk2"]

        retriever = self.embed_manager.run()
        mock_generate_embeddings.assert_called_once_with(["chunk1", "chunk2"], use_cache=False)  # Corrected assertion
        self.assertTrue(retriever)

        # Ensure the methods were called
        mock_load_documents.assert_called_once()
        mock_split_documents.assert_called_once_with(["doc1", "doc2"])
        mock_generate_embeddings.assert_called_once_with(["chunk1", "chunk2"],
                                                         use_cache=False)

if __name__ == "__main__":
    unittest.main()
