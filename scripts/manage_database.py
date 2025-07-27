#!/usr/bin/env python3
"""
Database Management CLI Utility

This script provides command-line utilities for managing the MLOps platform database,
including initialization, migrations, backups, and maintenance operations.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.api.config import APIConfig
from src.api.database_init import (
    initialize_database, reset_database, backup_database, get_database_info
)
from src.api.migrations import create_migrator
from src.api.database import get_database_manager


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def cmd_init(args):
    """Initialize database command."""
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Initializing database: {config.database_url}")
    
    success = initialize_database(
        config=config,
        run_migrations=not args.no_migrations,
        create_sample_data=args.sample_data
    )
    
    if success:
        print("✅ Database initialized successfully")
        return 0
    else:
        print("❌ Database initialization failed")
        return 1


def cmd_migrate(args):
    """Run database migrations command."""
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Running migrations on: {config.database_url}")
    
    migrator = create_migrator(config)
    
    if args.version:
        # Apply specific migration
        migration = None
        for m in migrator.migrations:
            if m["version"] == args.version:
                migration = m
                break
        
        if not migration:
            print(f"❌ Migration {args.version} not found")
            return 1
        
        success = migrator.apply_migration(migration)
    else:
        # Apply all pending migrations
        success = migrator.migrate_to_latest()
    
    if success:
        print("✅ Migrations completed successfully")
        return 0
    else:
        print("❌ Migration failed")
        return 1


def cmd_rollback(args):
    """Rollback migration command."""
    if not args.version:
        print("❌ --version required for rollback")
        return 1
    
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Rolling back migration {args.version} on: {config.database_url}")
    
    migrator = create_migrator(config)
    success = migrator.rollback_migration(args.version)
    
    if success:
        print("✅ Rollback completed successfully")
        return 0
    else:
        print("❌ Rollback failed")
        return 1


def cmd_status(args):
    """Show database status command."""
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Database Status: {config.database_url}")
    print("=" * 50)
    
    # Get database info
    info = get_database_info(config)
    
    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return 1
    
    # Health check
    health = info.get("health_check", False)
    print(f"Health Check: {'✅ Healthy' if health else '❌ Unhealthy'}")
    
    # Current version
    current_version = info.get("current_version")
    print(f"Current Version: {current_version or 'No migrations applied'}")
    
    # Applied migrations
    applied_migrations = info.get("applied_migrations", [])
    print(f"Applied Migrations: {len(applied_migrations)}")
    for migration in applied_migrations:
        print(f"  - {migration}")
    
    # Schema validation
    schema_validation = info.get("schema_validation", {})
    schema_valid = schema_validation.get("valid", False)
    print(f"Schema Valid: {'✅ Valid' if schema_valid else '❌ Invalid'}")
    
    if not schema_valid:
        missing_tables = schema_validation.get("missing_tables", [])
        if missing_tables:
            print(f"  Missing Tables: {missing_tables}")
        
        schema_issues = schema_validation.get("schema_issues", [])
        if schema_issues:
            print("  Schema Issues:")
            for issue in schema_issues:
                print(f"    - Table {issue['table']}: {issue}")
    
    # Prediction statistics
    prediction_stats = info.get("prediction_stats", {})
    if prediction_stats:
        print("\nPrediction Statistics:")
        print(f"  Total Predictions: {prediction_stats.get('total_predictions', 0)}")
        print(f"  Successful: {prediction_stats.get('successful_predictions', 0)}")
        print(f"  Failed: {prediction_stats.get('failed_predictions', 0)}")
        print(f"  Success Rate: {prediction_stats.get('success_rate', 0):.2%}")
        print(f"  Avg Processing Time: {prediction_stats.get('average_processing_time_ms', 0):.2f}ms")
    
    return 0


def cmd_backup(args):
    """Backup database command."""
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Creating backup of: {config.database_url}")
    
    backup_path = backup_database(config=config, backup_path=args.output)
    
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
        return 0
    else:
        print("❌ Backup failed")
        return 1


def cmd_reset(args):
    """Reset database command."""
    if not args.confirm:
        print("❌ This will delete ALL data! Use --confirm to proceed")
        return 1
    
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"⚠️  RESETTING DATABASE: {config.database_url}")
    print("This will delete ALL data!")
    
    success = reset_database(config=config, confirm=True)
    
    if success:
        print("✅ Database reset completed")
        return 0
    else:
        print("❌ Database reset failed")
        return 1


def cmd_cleanup(args):
    """Cleanup old records command."""
    config = APIConfig()
    if args.database_url:
        config.database_url = args.database_url
    
    print(f"Cleaning up records older than {args.days} days")
    
    db_manager = get_database_manager(config)
    deleted_count = db_manager.cleanup_old_records(days_to_keep=args.days)
    
    print(f"✅ Cleaned up {deleted_count} old records")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLOps Platform Database Management Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                          # Initialize database with migrations
  %(prog)s init --sample-data            # Initialize with sample data
  %(prog)s migrate                       # Run all pending migrations
  %(prog)s migrate --version 001_initial # Run specific migration
  %(prog)s rollback --version 002_batch  # Rollback specific migration
  %(prog)s status                        # Show database status
  %(prog)s backup                        # Create database backup
  %(prog)s backup --output backup.db     # Create backup with specific name
  %(prog)s reset --confirm               # Reset database (DESTRUCTIVE!)
  %(prog)s cleanup --days 30             # Clean up records older than 30 days
        """
    )
    
    # Global arguments
    parser.add_argument("--database-url", help="Database URL override")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument("--no-migrations", action="store_true",
                            help="Skip running migrations")
    init_parser.add_argument("--sample-data", action="store_true",
                            help="Create sample data for testing")
    init_parser.set_defaults(func=cmd_init)
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("--version", help="Specific migration version to apply")
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migration")
    rollback_parser.add_argument("--version", required=True,
                                help="Migration version to rollback")
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.set_defaults(func=cmd_status)
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument("--output", help="Backup file path")
    backup_parser.set_defaults(func=cmd_backup)
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database (DESTRUCTIVE!)")
    reset_parser.add_argument("--confirm", action="store_true",
                             help="Confirm destructive operation")
    reset_parser.set_defaults(func=cmd_reset)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old records")
    cleanup_parser.add_argument("--days", type=int, default=30,
                               help="Number of days of records to keep")
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Unexpected error occurred")
        return 1


if __name__ == "__main__":
    sys.exit(main())