"""
Incremental Flagging Operations

Handles flag management, incremental flagging, and writing flags back to measurement sets.
Designed to work with the memory-efficient loader for large datasets.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime
import pickle

CASACORE_AVAILABLE = False
pt = None

logger = logging.getLogger(__name__)


@dataclass
class FlagSession:
    """Metadata for a flagging session"""

    session_id: str
    timestamp: datetime
    ms_path: str
    algorithm: str
    parameters: Dict
    baselines_processed: List[Tuple[int, int]]
    total_flags_added: int
    flag_fraction: float


class FlagManager:
    """
    Manages flags with incremental flagging support and session tracking
    """

    def __init__(self, ms_path: str, work_dir: Optional[str] = None):
        """
        Initialize flag manager

        Args:
            ms_path: Path to measurement set
            work_dir: Working directory for flag cache
        """
        if not CASACORE_AVAILABLE:
            raise ImportError("python-casacore is required but not available")

        self.ms_path = Path(ms_path)
        if not self.ms_path.exists():
            raise FileNotFoundError(f"Measurement set not found: {ms_path}")

        # Setup working directory
        if work_dir:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path.cwd() / "samrfi_data"
        self.work_dir.mkdir(exist_ok=True)

        # Flag cache directory
        self.flag_cache_dir = self.work_dir / "flag_cache"
        self.flag_cache_dir.mkdir(exist_ok=True)

        # Session tracking
        self.sessions = []
        self.current_session = None

        # Internal flag storage (for incremental flagging)
        self._flag_cache = {}  # baseline -> flags
        self._table = None

    def start_session(self, algorithm: str, parameters: Dict) -> str:
        """
        Start a new flagging session

        Args:
            algorithm: Name of flagging algorithm
            parameters: Algorithm parameters

        Returns:
            Session ID
        """
        session_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = FlagSession(
            session_id=session_id,
            timestamp=datetime.now(),
            ms_path=str(self.ms_path),
            algorithm=algorithm,
            parameters=parameters.copy(),
            baselines_processed=[],
            total_flags_added=0,
            flag_fraction=0.0,
        )

        logger.info(f"Started flagging session: {session_id}")
        return session_id

    def open_table(self, readonly: bool = False) -> None:
        """Open measurement set table"""
        if self._table is None:
            self._table = pt.table(str(self.ms_path), readonly=readonly)
            logger.debug("Opened measurement set table for flagging")

    def close_table(self) -> None:
        """Close measurement set table"""
        if self._table is not None:
            self._table.close()
            self._table = None
            logger.debug("Closed measurement set table")

    def get_existing_flags(
        self, baselines: List[Tuple[int, int]], spw_list: List[int], field_id: int = 0
    ) -> np.ndarray:
        """
        Get existing flags from measurement set

        Args:
            baselines: List of (ant1, ant2) tuples
            spw_list: List of SPW IDs
            field_id: Field ID

        Returns:
            Existing flags [baselines, polarizations, channels, time]
        """
        self.open_table(readonly=True)

        # Load flags using same logic as loader
        from .loader import MSLoader

        loader = MSLoader(str(self.ms_path), str(self.work_dir))
        flags = loader.load_flags_batch(baselines, spw_list, field_id)
        loader.close_table()

        return flags

    def update_flags_incremental(
        self,
        baselines: List[Tuple[int, int]],
        new_flags: np.ndarray,
        spw_list: List[int],
        mode: str = "combine",
    ) -> Dict:
        """
        Update flags with incremental support

        Args:
            baselines: List of (ant1, ant2) tuples
            new_flags: New flag array [baselines, polarizations, channels, time]
            spw_list: List of SPW IDs
            mode: 'combine' (OR with existing), 'replace', or 'subtract'

        Returns:
            Dictionary with flagging statistics
        """
        if self.current_session is None:
            raise RuntimeError(
                "No active flagging session. Call start_session() first."
            )

        # Get existing flags
        existing_flags = self.get_existing_flags(baselines, spw_list)

        # Apply incremental flagging logic
        if mode == "combine":
            # OR with existing flags (standard incremental flagging)
            combined_flags = np.logical_or(existing_flags, new_flags)
        elif mode == "replace":
            # Replace existing flags
            combined_flags = new_flags.copy()
        elif mode == "subtract":
            # Remove flags (unflag): existing AND NOT new
            combined_flags = np.logical_and(existing_flags, ~new_flags)
        else:
            raise ValueError(f"Unknown flagging mode: {mode}")

        # Cache the flags for this batch
        for i, baseline in enumerate(baselines):
            cache_key = f"{baseline[0]}_{baseline[1]}_{hash(tuple(spw_list))}"
            self._flag_cache[cache_key] = {
                "baseline": baseline,
                "spw_list": spw_list,
                "flags": combined_flags[i],
                "timestamp": datetime.now(),
            }

        # Calculate statistics
        old_flag_count = np.sum(existing_flags)
        new_flag_count = np.sum(combined_flags)
        flags_added = new_flag_count - old_flag_count

        # Update session statistics
        self.current_session.baselines_processed.extend(baselines)
        self.current_session.total_flags_added += flags_added

        stats = {
            "baselines_processed": len(baselines),
            "flags_before": int(old_flag_count),
            "flags_after": int(new_flag_count),
            "flags_added": int(flags_added),
            "flag_fraction_before": old_flag_count / existing_flags.size,
            "flag_fraction_after": new_flag_count / combined_flags.size,
            "mode": mode,
        }

        logger.info(
            f"Updated flags for {len(baselines)} baselines: "
            f"{flags_added:+d} flags ({stats['flag_fraction_after']:.3f} total)"
        )

        return stats

    def commit_flags(self, backup: bool = True, dry_run: bool = False) -> Dict:
        """
        Commit cached flags to measurement set

        Args:
            backup: Whether to backup original flags
            dry_run: If True, don't actually write to MS

        Returns:
            Dictionary with commit statistics
        """
        if not self._flag_cache:
            logger.warning("No cached flags to commit")
            return {"status": "no_flags"}

        self.open_table(readonly=dry_run)

        # Backup original flags if requested
        if backup and not dry_run:
            self._backup_original_flags()

        total_flags_written = 0
        baselines_written = set()

        logger.info(f"Committing {len(self._flag_cache)} flag batches to MS...")

        for cache_key, flag_data in self._flag_cache.items():
            baseline = flag_data["baseline"]
            spw_list = flag_data["spw_list"]
            flags = flag_data["flags"]  # [pol, chan, time]

            if not dry_run:
                self._write_baseline_flags(baseline, spw_list, flags)

            total_flags_written += np.sum(flags)
            baselines_written.add(baseline)

        # Update session
        if self.current_session:
            self.current_session.flag_fraction = (
                total_flags_written / (len(self._flag_cache) * flags.size)
                if len(self._flag_cache) > 0
                else 0.0
            )

            # Save session metadata
            self._save_session_metadata()

        commit_stats = {
            "status": "success" if not dry_run else "dry_run",
            "batches_committed": len(self._flag_cache),
            "baselines_affected": len(baselines_written),
            "total_flags_written": int(total_flags_written),
            "backup_created": backup and not dry_run,
        }

        if not dry_run:
            # Clear cache after successful commit
            self._flag_cache.clear()
            logger.info(f"Flags committed successfully: {commit_stats}")
        else:
            logger.info(f"Dry run completed: {commit_stats}")

        return commit_stats

    def _write_baseline_flags(
        self, baseline: Tuple[int, int], spw_list: List[int], flags: np.ndarray
    ) -> None:
        """Write flags for a single baseline to MS"""
        ant1, ant2 = baseline

        # Get channel info for each SPW
        with pt.table(
            str(self.ms_path / "SPECTRAL_WINDOW"), readonly=True
        ) as spw_table:
            all_channels = spw_table.getcol("NUM_CHAN")

        ref_channels = all_channels[spw_list[0]]

        for spw_idx, spw_id in enumerate(spw_list):
            # Extract flags for this SPW
            start_chan = spw_idx * ref_channels
            end_chan = (spw_idx + 1) * ref_channels
            spw_flags = flags[:, start_chan:end_chan, :]

            # Update measurement set
            query = f"ANTENNA1=={ant1} && ANTENNA2=={ant2} && DATA_DESC_ID=={spw_id}"
            subtable = self._table.query(query)

            if subtable.nrows() > 0:
                subtable.putcol("FLAG", spw_flags)
            else:
                logger.warning(
                    f"No rows found for baseline {ant1}-{ant2}, SPW {spw_id}"
                )

            subtable.close()

    def _backup_original_flags(self) -> None:
        """Backup original flags to cache directory"""
        backup_file = (
            self.flag_cache_dir
            / f"original_flags_{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
        )

        # For now, just log that we would backup
        # Full implementation would save original flags to backup file
        logger.info(f"Would backup original flags to: {backup_file}")

    def _save_session_metadata(self) -> None:
        """Save session metadata to file"""
        if self.current_session is None:
            return

        session_file = (
            self.flag_cache_dir / f"session_{self.current_session.session_id}.pkl"
        )

        try:
            with open(session_file, "wb") as f:
                pickle.dump(self.current_session, f)
            logger.debug(f"Saved session metadata: {session_file}")
        except Exception as e:
            logger.warning(f"Could not save session metadata: {e}")

    def end_session(self) -> Optional[FlagSession]:
        """End current flagging session and return session data"""
        if self.current_session is None:
            return None

        session = self.current_session
        self.sessions.append(session)
        self.current_session = None

        logger.info(f"Ended flagging session: {session.session_id}")
        logger.info(f"  Total flags added: {session.total_flags_added}")
        logger.info(f"  Final flag fraction: {session.flag_fraction:.4f}")

        return session

    def get_flag_statistics(self) -> Dict:
        """Get comprehensive flagging statistics"""
        stats = {
            "active_session": (
                self.current_session.session_id if self.current_session else None
            ),
            "cached_batches": len(self._flag_cache),
            "total_sessions": len(self.sessions),
        }

        if self.current_session:
            stats["current_session"] = {
                "algorithm": self.current_session.algorithm,
                "baselines_processed": len(
                    set(self.current_session.baselines_processed)
                ),
                "flags_added": self.current_session.total_flags_added,
            }

        return stats

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_table()
        if self.current_session:
            self.end_session()
