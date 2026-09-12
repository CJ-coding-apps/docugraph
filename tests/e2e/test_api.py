"""End-to-end tests for the REST API."""

import pytest
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def client(temp_data_dir, monkeypatch):
    """Create a test client with isolated data directory."""
    monkeypatch.setenv("DOCUGRAPH_STORAGE__DATA_DIR", temp_data_dir)

    # Import after setting env var
    from docugraph.interfaces.api.main import app

    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and stats endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        response = client.get("/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_chunks" in data
        assert "data_directory" in data
        assert "embedding_model" in data


class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_empty_index(self, client):
        """Test search on empty index."""
        response = client.post(
            "/v1/search",
            json={"query": "test query"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "test query"
        assert "results" in data
        assert "total" in data
        assert "took_ms" in data

    def test_search_with_top_k(self, client):
        """Test search with custom top_k."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "top_k": 3},
        )
        assert response.status_code == 200

    def test_search_missing_query(self, client):
        """Test search with missing query."""
        response = client.post("/v1/search", json={})
        assert response.status_code == 422  # Validation error

    def test_hybrid_search(self, client):
        """Test hybrid search endpoint."""
        response = client.post(
            "/v1/search/hybrid",
            json={
                "query": "test query",
                "mode": "hybrid",
                "top_k": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "test query"
        assert "results" in data


class TestMemoryEndpoints:
    """Tests for memory endpoints."""

    def test_memory_set_and_get(self, client):
        """Test memory set and get endpoints."""
        # Set a value
        response = client.post(
            "/v1/memory",
            json={
                "key": "test_key",
                "value": "test_value",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["key"] == "test_key"
        assert data["value"] == "test_value"

        # Get the value
        response = client.get("/v1/memory/test_key")
        assert response.status_code == 200

        data = response.json()
        assert data["value"] == "test_value"

    def test_memory_get_nonexistent(self, client):
        """Test getting nonexistent memory key."""
        response = client.get("/v1/memory/nonexistent_key")
        assert response.status_code == 404

    def test_memory_list(self, client):
        """Test memory list endpoint."""
        # Set some values
        client.post("/v1/memory", json={"key": "key1", "value": "value1"})
        client.post("/v1/memory", json={"key": "key2", "value": "value2"})

        # List keys
        response = client.get("/v1/memory")
        assert response.status_code == 200

        data = response.json()
        assert "keys" in data
        assert "key1" in data["keys"]
        assert "key2" in data["keys"]

    def test_memory_delete(self, client):
        """Test memory delete endpoint."""
        # Set a value
        client.post("/v1/memory", json={"key": "to_delete", "value": "value"})

        # Delete it
        response = client.delete("/v1/memory/to_delete")
        assert response.status_code == 200

        data = response.json()
        assert data["deleted"] is True

        # Verify deletion
        response = client.get("/v1/memory/to_delete")
        assert response.status_code == 404

    def test_memory_with_scoping(self, client):
        """Test memory with repository/branch scoping."""
        # Set value with custom scope
        response = client.post(
            "/v1/memory",
            json={
                "key": "scoped_key",
                "value": "scoped_value",
                "repository": "test-repo",
                "branch": "feature",
            },
        )
        assert response.status_code == 200

        # Get with scope
        response = client.get(
            "/v1/memory/scoped_key",
            params={"repository": "test-repo", "branch": "feature"},
        )
        assert response.status_code == 200
        assert response.json()["value"] == "scoped_value"

        # Same key, different scope should not exist
        response = client.get(
            "/v1/memory/scoped_key",
            params={"repository": "other-repo", "branch": "main"},
        )
        assert response.status_code == 404


class TestIndexEndpoints:
    """Tests for indexing endpoints."""

    def test_index_local_missing_path(self, client):
        """Test index-local with missing path."""
        response = client.post(
            "/v1/index/local",
            json={"path": "/nonexistent/path"},
        )
        assert response.status_code == 400

    def test_clear_index(self, client):
        """Test clearing the index."""
        response = client.delete("/v1/index")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"


class TestRepoEndpoints:
    """Tests for repository endpoints."""

    def test_list_repos_empty(self, client):
        """Test listing repos when none are indexed."""
        response = client.get("/v1/repos")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
