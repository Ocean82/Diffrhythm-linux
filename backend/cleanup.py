"""
File cleanup and retention module
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from backend.config import Config

logger = logging.getLogger(__name__)

# Import S3 functions if available
try:
    from backend.s3_storage import delete_from_s3, file_exists_in_s3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger.warning("S3 storage module not available, cleanup will only handle local files")


def get_files_to_delete() -> List[Dict[str, Any]]:
    """
    Find files that exceed the retention period
    
    Returns:
        List of dicts with job_id, file_path, age_days, and s3_url if applicable
    """
    if not Config.CLEANUP_ENABLED:
        return []
    
    files_to_delete = []
    retention_days = Config.FILE_RETENTION_DAYS
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
    
    # Scan output directory
    if not Config.OUTPUT_DIR.exists():
        return []
    
    for job_dir in Config.OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        
        job_id = job_dir.name
        
        # Find audio files in job directory
        audio_files = list(job_dir.glob("*.wav"))
        if not audio_files:
            continue
        
        # Check file modification time
        for audio_file in audio_files:
            file_mtime = datetime.fromtimestamp(audio_file.stat().st_mtime)
            age_days = (datetime.utcnow() - file_mtime).days
            
            if file_mtime < cutoff_date:
                s3_url = None
                if Config.S3_ENABLED and S3_AVAILABLE:
                    # Check if file exists in S3
                    if file_exists_in_s3(job_id, audio_file.name):
                        s3_url = f"s3://{Config.S3_BUCKET}/{Config.S3_PREFIX.rstrip('/')}/{job_id}/{audio_file.name}"
                
                files_to_delete.append({
                    "job_id": job_id,
                    "file_path": audio_file,
                    "age_days": age_days,
                    "s3_url": s3_url
                })
    
    return files_to_delete


def cleanup_old_files() -> Dict[str, int]:
    """
    Delete files older than retention period
    
    Returns:
        Dict with counts of deleted files
    """
    if not Config.CLEANUP_ENABLED:
        logger.info("Cleanup is disabled")
        return {"deleted": 0, "errors": 0}
    
    files_to_delete = get_files_to_delete()
    
    if not files_to_delete:
        logger.info("No files to clean up")
        return {"deleted": 0, "errors": 0}
    
    deleted_count = 0
    error_count = 0
    
    logger.info(f"Starting cleanup: {len(files_to_delete)} files to delete")
    
    for file_info in files_to_delete:
        job_id = file_info["job_id"]
        file_path = file_info["file_path"]
        age_days = file_info["age_days"]
        s3_url = file_info.get("s3_url")
        
        try:
            # Delete from S3 if applicable
            if s3_url and Config.S3_ENABLED and S3_AVAILABLE:
                try:
                    delete_from_s3(job_id, file_path.name)
                    logger.info(f"Deleted from S3: {s3_url}")
                except Exception as e:
                    logger.warning(f"Failed to delete from S3 {s3_url}: {e}")
            
            # Delete local file
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted local file: {file_path} (age: {age_days} days)")
            
            # Delete job directory if empty
            job_dir = file_path.parent
            if job_dir.exists() and not any(job_dir.iterdir()):
                job_dir.rmdir()
                logger.info(f"Removed empty job directory: {job_dir}")
            
            deleted_count += 1
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
            error_count += 1
    
    logger.info(f"Cleanup complete: {deleted_count} files deleted, {error_count} errors")
    
    return {
        "deleted": deleted_count,
        "errors": error_count,
        "total": len(files_to_delete)
    }


def cleanup_old_jobs(job_manager=None) -> int:
    """
    Remove old job records from memory (optional, for memory management)
    
    Note: This only removes from in-memory storage. For persistent storage,
    jobs should be archived rather than deleted for audit purposes.
    
    Args:
        job_manager: JobManager instance (optional)
        
    Returns:
        Number of jobs removed
    """
    if not Config.CLEANUP_ENABLED or not job_manager:
        return 0
    
    retention_days = Config.FILE_RETENTION_DAYS
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
    
    removed_count = 0
    
    # Get list of jobs to check
    job_ids_to_check = list(job_manager.jobs.keys())
    
    for job_id in job_ids_to_check:
        job = job_manager.jobs.get(job_id)
        if not job:
            continue
        
        # Only remove completed or failed jobs that are old
        if job.get("status") not in ["completed", "failed"]:
            continue
        
        completed_at = job.get("completed_at")
        if not completed_at:
            continue
        
        try:
            # Parse ISO format timestamp
            if isinstance(completed_at, str):
                if completed_at.endswith("Z"):
                    completed_at = completed_at[:-1]
                completed_date = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            else:
                completed_date = completed_at
            
            # Remove timezone for comparison
            if completed_date.tzinfo:
                completed_date = completed_date.replace(tzinfo=None)
            
            if completed_date < cutoff_date:
                # Check if file still exists (should have been deleted by cleanup_old_files)
                output_file = job.get("output_file")
                if output_file:
                    file_path = Config.OUTPUT_DIR / output_file
                    if file_path.exists():
                        # File still exists, don't remove job record yet
                        continue
                
                # Remove job record
                del job_manager.jobs[job_id]
                removed_count += 1
                logger.debug(f"Removed old job record: {job_id}")
                
        except Exception as e:
            logger.warning(f"Error processing job {job_id} for cleanup: {e}")
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} old job records from memory")
    
    return removed_count
