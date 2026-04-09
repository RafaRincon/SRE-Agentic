"""
Unit Tests: Hybrid Search logic in db_provider
Tests function signatures, fallback behavior, and query construction.
"""
import sys, asyncio, unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.providers import db_provider


class TestVectorSearchSignature(unittest.TestCase):
    """Test that vector_search accepts the new query_text parameter."""

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_accepts_query_text(self, mock_settings, mock_container):
        """vector_search should accept query_text as a keyword arg."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")
        mock_container.return_value.query_items.return_value = []

        # Should not raise TypeError
        result = db_provider.vector_search(
            query_vector=[0.1] * 768,
            query_text="test query",
            top_k=3,
        )
        self.assertIsInstance(result, list)

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_works_without_query_text(self, mock_settings, mock_container):
        """vector_search should work without query_text (backward compat)."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")
        mock_container.return_value.query_items.return_value = []

        result = db_provider.vector_search(
            query_vector=[0.1] * 768,
            top_k=3,
        )
        self.assertIsInstance(result, list)


class TestHybridFallback(unittest.TestCase):
    """Test that hybrid search falls back to vector-only on failure."""

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_fallback_when_hybrid_fails(self, mock_settings, mock_container):
        """If RRF query fails, should fallback to vector-only."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")

        container_mock = MagicMock()
        mock_container.return_value = container_mock

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (hybrid) fails
                raise Exception("FullTextScore not supported")
            else:
                # Second call (vector-only) succeeds
                return [{"id": "test", "chunk_text": "fallback result"}]

        container_mock.query_items.side_effect = side_effect

        result = db_provider.vector_search(
            query_vector=[0.1] * 768,
            query_text="test",
            top_k=3,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "test")
        # query_items should be called twice: hybrid (fails) + vector-only (succeeds)
        self.assertEqual(call_count, 2)

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_no_fallback_when_no_query_text(self, mock_settings, mock_container):
        """Without query_text, should go straight to vector-only."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")
        container_mock = MagicMock()
        mock_container.return_value = container_mock
        container_mock.query_items.return_value = [{"id": "v1"}]

        result = db_provider.vector_search(
            query_vector=[0.1] * 768,
            top_k=3,
        )
        self.assertEqual(len(result), 1)
        # Should only be called once (vector-only, no hybrid attempt)
        container_mock.query_items.assert_called_once()


class TestKnowledgeSearchSignature(unittest.TestCase):
    """Test that knowledge_search accepts the new query_text parameter."""

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_accepts_query_text(self, mock_settings, mock_container):
        mock_settings.return_value = MagicMock(cosmos_container_knowledge="sre_knowledge")
        mock_container.return_value.query_items.return_value = []

        result = db_provider.knowledge_search(
            query_vector=[0.1] * 768,
            query_text="runbook escalation",
            top_k=3,
        )
        self.assertIsInstance(result, list)


class TestHybridQueryConstruction(unittest.TestCase):
    """Test that the hybrid query contains the correct RRF syntax."""

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_hybrid_query_contains_rrf(self, mock_settings, mock_container):
        """The hybrid query should use ORDER BY RANK RRF(...)."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")
        container_mock = MagicMock()
        mock_container.return_value = container_mock
        container_mock.query_items.return_value = []

        db_provider.vector_search(
            query_vector=[0.1] * 768,
            query_text="test query",
            top_k=3,
        )

        # Check the first call was the hybrid query
        call_args = container_mock.query_items.call_args
        query = call_args.kwargs.get("query", "")
        self.assertIn("RANK RRF", query)
        self.assertIn("VectorDistance", query)
        self.assertIn("FullTextScore", query)

    @patch.object(db_provider, 'get_container')
    @patch.object(db_provider, 'get_settings')
    def test_vector_only_query_no_rrf(self, mock_settings, mock_container):
        """Without query_text, query should NOT contain RRF/FullTextScore."""
        mock_settings.return_value = MagicMock(cosmos_container_chunks="eshop_chunks")
        container_mock = MagicMock()
        mock_container.return_value = container_mock
        container_mock.query_items.return_value = []

        db_provider.vector_search(
            query_vector=[0.1] * 768,
            top_k=3,
        )

        call_args = container_mock.query_items.call_args
        query = call_args.kwargs.get("query", "")
        self.assertNotIn("FullTextScore", query)
        self.assertIn("VectorDistance", query)


if __name__ == "__main__":
    unittest.main(verbosity=2)
