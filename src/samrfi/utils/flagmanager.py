"""
CASA flagmanager utility wrappers for managing multiple flag versions.

The flagmanager allows saving, restoring, and managing different versions of the
FLAG column in a measurement set. This is essential for comparing different
flagging methods without permanently modifying the data.

Typical workflow:
    1. Save original flags: save_flag_version(ms, 'original')
    2. Run method 1 → save: save_flag_version(ms, 'tfcrop')
    3. Restore original: restore_flag_version(ms, 'original')
    4. Run method 2 → save: save_flag_version(ms, 'sam_rfi')
    5. Compare results
"""

import logging

try:
    from casatasks import flagmanager

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False
    logging.warning(
        "CASA tasks not available. Flagmanager functions will not work. "
        "Install with: pip install casatasks"
    )

logger = logging.getLogger(__name__)


def save_flag_version(
    ms_path: str, version_name: str, comment: str = "", merge: str = "replace"
) -> bool:
    """
    Save current FLAG column state as a named version.

    Args:
        ms_path: Path to measurement set
        version_name: Name for this flag version (e.g., 'original', 'tfcrop', 'sam_rfi')
        comment: Optional description of this flag version
        merge: Merge mode - 'replace' (default) or 'or' or 'and'

    Returns:
        bool: True if successful, False otherwise

    Raises:
        RuntimeError: If CASA is not available
        Exception: If flagmanager fails

    Example:
        >>> save_flag_version('observation.ms', 'original', 'Flags before processing')
        ✓ Saved flag version: original
        True
    """
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA flagmanager not available. Install with: pip install casatasks")

    try:
        flagmanager(
            vis=ms_path, mode="save", versionname=version_name, comment=comment, merge=merge
        )
        logger.info(f"✓ Saved flag version: {version_name}")
        print(f"✓ Saved flag version: {version_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save flag version '{version_name}': {e}")
        print(f"✗ Failed to save flag version '{version_name}': {e}")
        return False


def restore_flag_version(ms_path: str, version_name: str, merge: str = "replace") -> bool:
    """
    Restore FLAG column to a previously saved version.

    Args:
        ms_path: Path to measurement set
        version_name: Name of version to restore
        merge: Merge mode - 'replace' (default) or 'or' or 'and'

    Returns:
        bool: True if successful, False otherwise

    Raises:
        RuntimeError: If CASA is not available
        Exception: If flagmanager fails

    Example:
        >>> restore_flag_version('observation.ms', 'original')
        ✓ Restored flag version: original
        True
    """
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA flagmanager not available. Install with: pip install casatasks")

    try:
        flagmanager(vis=ms_path, mode="restore", versionname=version_name, merge=merge)
        logger.info(f"✓ Restored flag version: {version_name}")
        print(f"✓ Restored flag version: {version_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to restore flag version '{version_name}': {e}")
        print(f"✗ Failed to restore flag version '{version_name}': {e}")
        return False


def list_flag_versions(ms_path: str) -> dict:
    """
    List all saved flag versions for a measurement set.

    Args:
        ms_path: Path to measurement set

    Returns:
        dict: Version metadata from CASA flagmanager

    Raises:
        RuntimeError: If CASA is not available

    Example:
        >>> versions = list_flag_versions('observation.ms')
        >>> print(versions.keys())
        dict_keys(['main', 'original', 'tfcrop', 'sam_rfi'])
    """
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA flagmanager not available. Install with: pip install casatasks")

    try:
        result = flagmanager(vis=ms_path, mode="list")
        logger.info(f"Listed {len(result)} flag versions for {ms_path}")
        return result
    except Exception as e:
        logger.error(f"✗ Failed to list flag versions: {e}")
        return {}


def delete_flag_version(ms_path: str, version_name: str) -> bool:
    """
    Delete a saved flag version.

    Args:
        ms_path: Path to measurement set
        version_name: Name of version to delete

    Returns:
        bool: True if successful, False otherwise

    Raises:
        RuntimeError: If CASA is not available

    Example:
        >>> delete_flag_version('observation.ms', 'test_version')
        ✓ Deleted flag version: test_version
        True
    """
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA flagmanager not available. Install with: pip install casatasks")

    try:
        flagmanager(vis=ms_path, mode="delete", versionname=version_name)
        logger.info(f"✓ Deleted flag version: {version_name}")
        print(f"✓ Deleted flag version: {version_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to delete flag version '{version_name}': {e}")
        print(f"✗ Failed to delete flag version '{version_name}': {e}")
        return False


def rename_flag_version(ms_path: str, old_name: str, new_name: str, comment: str = "") -> bool:
    """
    Rename a saved flag version.

    Args:
        ms_path: Path to measurement set
        old_name: Current version name
        new_name: New version name
        comment: Optional new comment

    Returns:
        bool: True if successful, False otherwise

    Raises:
        RuntimeError: If CASA is not available

    Example:
        >>> rename_flag_version('observation.ms', 'temp', 'final_flags')
        ✓ Renamed flag version: temp → final_flags
        True
    """
    if not CASA_AVAILABLE:
        raise RuntimeError("CASA flagmanager not available. Install with: pip install casatasks")

    try:
        flagmanager(
            vis=ms_path, mode="rename", oldname=old_name, versionname=new_name, comment=comment
        )
        logger.info(f"✓ Renamed flag version: {old_name} → {new_name}")
        print(f"✓ Renamed flag version: {old_name} → {new_name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to rename flag version '{old_name}': {e}")
        print(f"✗ Failed to rename flag version '{old_name}': {e}")
        return False


class FlagVersionContext:
    """
    Context manager for temporarily restoring a flag version.

    Saves current flags, restores specified version, and restores original on exit.

    Example:
        >>> with FlagVersionContext('observation.ms', 'original'):
        ...     # Work with original flags
        ...     run_analysis()
        ... # Automatically restored to state before context
    """

    def __init__(self, ms_path: str, version_name: str):
        self.ms_path = ms_path
        self.version_name = version_name
        self.temp_version = f"_temp_{id(self)}"

    def __enter__(self):
        """Save current state and restore specified version."""
        save_flag_version(self.ms_path, self.temp_version, comment="Temporary save")
        restore_flag_version(self.ms_path, self.version_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state and cleanup."""
        restore_flag_version(self.ms_path, self.temp_version)
        delete_flag_version(self.ms_path, self.temp_version)
        return False  # Don't suppress exceptions
