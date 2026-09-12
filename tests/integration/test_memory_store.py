"""Integration tests for MemoryStore."""

import pytest
import tempfile
from pathlib import Path

from docugraph.storage.memory_store import MemoryStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        yield str(db_path)


class TestMemoryStore:
    """Integration tests for MemoryStore."""

    def test_store_and_retrieve(self, temp_db):
        """Test basic store and retrieve operations."""
        store = MemoryStore(db_path=temp_db)

        # Store a value
        entry = store.set("test_key", "test_value")
        assert entry.key == "test_key"
        assert entry.value == "test_value"

        # Retrieve it
        value = store.get("test_key")
        assert value == "test_value"

    def test_store_complex_value(self, temp_db):
        """Test storing complex values (dict, list)."""
        store = MemoryStore(db_path=temp_db)

        # Store a dict
        data = {"name": "test", "items": [1, 2, 3], "nested": {"key": "value"}}
        store.set("complex", data)

        # Retrieve and verify
        retrieved = store.get("complex")
        assert retrieved == data

    def test_repository_branch_scoping(self, temp_db):
        """Test that values are scoped by repository and branch."""
        store = MemoryStore(db_path=temp_db)

        # Store same key in different scopes
        store.set("config", "value1", repository="repo1", branch="main")
        store.set("config", "value2", repository="repo1", branch="dev")
        store.set("config", "value3", repository="repo2", branch="main")

        # Retrieve and verify isolation
        assert store.get("config", repository="repo1", branch="main") == "value1"
        assert store.get("config", repository="repo1", branch="dev") == "value2"
        assert store.get("config", repository="repo2", branch="main") == "value3"

    def test_list_keys(self, temp_db):
        """Test listing keys in a scope."""
        store = MemoryStore(db_path=temp_db)

        # Store multiple keys
        store.set("key1", "value1", repository="test", branch="main")
        store.set("key2", "value2", repository="test", branch="main")
        store.set("key3", "value3", repository="test", branch="dev")

        # List keys
        main_keys = store.list_keys(repository="test", branch="main")
        assert "key1" in main_keys
        assert "key2" in main_keys
        assert "key3" not in main_keys

        dev_keys = store.list_keys(repository="test", branch="dev")
        assert "key3" in dev_keys
        assert "key1" not in dev_keys

    def test_delete_key(self, temp_db):
        """Test deleting a key."""
        store = MemoryStore(db_path=temp_db)

        # Store and verify
        store.set("to_delete", "value")
        assert store.get("to_delete") is not None

        # Delete
        deleted = store.delete("to_delete")
        assert deleted is True

        # Verify deletion
        assert store.get("to_delete") is None

    def test_delete_nonexistent_key(self, temp_db):
        """Test deleting a key that doesn't exist."""
        store = MemoryStore(db_path=temp_db)
        deleted = store.delete("nonexistent")
        assert deleted is False

    def test_update_existing_key(self, temp_db):
        """Test updating an existing key."""
        store = MemoryStore(db_path=temp_db)

        # Store initial value
        store.set("update_me", "initial")
        assert store.get("update_me") == "initial"

        # Update
        store.set("update_me", "updated")
        assert store.get("update_me") == "updated"

    def test_get_nonexistent_key(self, temp_db):
        """Test getting a key that doesn't exist."""
        store = MemoryStore(db_path=temp_db)
        value = store.get("nonexistent")
        assert value is None

    def test_persistence(self, temp_db):
        """Test that data persists across store instances."""
        # Store data
        store1 = MemoryStore(db_path=temp_db)
        store1.set("persistent", "data")

        # Create new instance and verify
        store2 = MemoryStore(db_path=temp_db)
        assert store2.get("persistent") == "data"

    def test_timestamps(self, temp_db):
        """Test that timestamps are set correctly."""
        store = MemoryStore(db_path=temp_db)

        entry = store.set("timestamped", "value")
        assert entry.created_at is not None
        assert entry.accessed_at is not None

    def test_clear_by_scope(self, temp_db):
        """Test clearing all entries in a scope."""
        store = MemoryStore(db_path=temp_db)

        # Store in multiple scopes
        store.set("key1", "value1", repository="repo1", branch="main")
        store.set("key2", "value2", repository="repo1", branch="main")
        store.set("key3", "value3", repository="repo2", branch="main")

        # Clear repo1/main
        store.clear(repository="repo1", branch="main")

        # Verify
        assert store.get("key1", repository="repo1", branch="main") is None
        assert store.get("key2", repository="repo1", branch="main") is None
        assert store.get("key3", repository="repo2", branch="main") == "value3"
