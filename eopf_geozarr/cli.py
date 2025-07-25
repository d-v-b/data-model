#!/usr/bin/env python3
"""
Command-line interface for eopf-geozarr.

This module provides CLI commands for converting EOPF datasets to GeoZarr compliant format.
"""

import argparse
import sys
from pathlib import Path

import xarray as xr

from . import create_geozarr_dataset
from .conversion import is_s3_path, validate_s3_access, get_s3_credentials_info


def convert_command(args: argparse.Namespace) -> None:
    """
    Convert EOPF dataset to GeoZarr compliant format.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Validate input path (handle both local paths and URLs)
    input_path_str = args.input_path
    if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        # URL - no local validation needed
        input_path = input_path_str
    else:
        # Local path - validate existence
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
        input_path = str(input_path)

    # Handle output path validation
    output_path_str = args.output_path
    if is_s3_path(output_path_str):
        # S3 path - validate S3 access
        print("🔍 Validating S3 access...")
        success, error_msg = validate_s3_access(output_path_str)
        if not success:
            print(f"❌ Error: Cannot access S3 path {output_path_str}")
            print(f"   Reason: {error_msg}")
            print("\n💡 S3 Configuration Help:")
            print("   Make sure you have S3 credentials configured:")
            print("   - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            print("   - Set AWS_DEFAULT_REGION (default: us-east-1)")
            print("   - For custom S3 providers (e.g., OVH Cloud), set AWS_S3_ENDPOINT")
            print("   - Or configure AWS CLI with 'aws configure'")
            print("   - Or use IAM roles if running on EC2")
            
            if args.verbose:
                creds_info = get_s3_credentials_info()
                print(f"\n🔧 Current AWS configuration:")
                for key, value in creds_info.items():
                    print(f"   {key}: {value or 'Not set'}")
            
            sys.exit(1)
        
        print("✅ S3 access validated successfully")
        output_path = output_path_str
    else:
        # Local path - create directory if it doesn't exist
        output_path = Path(output_path_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = str(output_path)

    if args.verbose:
        print(f"Loading EOPF dataset from: {input_path}")
        print(f"Groups to convert: {args.groups}")
        print(f"Output path: {output_path}")
        print(f"Spatial chunk size: {args.spatial_chunk}")
        print(f"Min dimension: {args.min_dimension}")
        print(f"Tile width: {args.tile_width}")

    try:
        # Load the EOPF DataTree
        print("Loading EOPF dataset...")
        dt = xr.open_datatree(str(input_path), engine="zarr")

        if args.verbose:
            print(f"Loaded DataTree with {len(dt.children)} groups")
            print("Available groups:")
            for group_name in dt.children:
                print(f"  - {group_name}")

        # Convert to GeoZarr compliant format
        print("Converting to GeoZarr compliant format...")
        dt_geozarr = create_geozarr_dataset(
            dt_input=dt,
            groups=args.groups,
            output_path=output_path,
            spatial_chunk=args.spatial_chunk,
            min_dimension=args.min_dimension,
            tile_width=args.tile_width,
            max_retries=args.max_retries,
        )

        print("✅ Successfully converted EOPF dataset to GeoZarr format")
        print(f"Output saved to: {output_path}")

        if args.verbose:
            print(f"Converted DataTree has {len(dt_geozarr.children)} groups")
            print("Converted groups:")
            for group_name in dt_geozarr.children:
                print(f"  - {group_name}")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def info_command(args: argparse.Namespace) -> None:
    """
    Display information about an EOPF dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Handle both local paths and URLs
    input_path_str = args.input_path
    if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        # URL - no local validation needed
        input_path = input_path_str
    else:
        # Local path - validate existence
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
        input_path = str(input_path)

    try:
        print(f"Loading dataset from: {input_path}")
        dt = xr.open_datatree(input_path, engine="zarr")

        print("\nDataset Information:")
        print("==================")
        print(f"Total groups: {len(dt.children)}")

        print("\nGroup structure:")
        print(dt)

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def validate_command(args: argparse.Namespace) -> None:
    """
    Validate GeoZarr compliance of a dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Handle both local paths and URLs
    input_path_str = args.input_path
    if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        # URL - no local validation needed
        input_path = input_path_str
    else:
        # Local path - validate existence
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
        input_path = str(input_path)

    try:
        print(f"Validating GeoZarr compliance for: {input_path}")
        dt = xr.open_datatree(input_path, engine="zarr")

        compliance_issues = []
        total_variables = 0
        compliant_variables = 0

        print("\nValidation Results:")
        print("==================")

        for group_name, group in dt.children.items():
            print(f"\nGroup: {group_name}")

            if not hasattr(group, "data_vars") or not group.data_vars:
                print("  ⚠️  No data variables found")
                continue

            for var_name, var in group.data_vars.items():
                total_variables += 1
                issues = []

                # Check for _ARRAY_DIMENSIONS
                if "_ARRAY_DIMENSIONS" not in var.attrs:
                    issues.append("Missing _ARRAY_DIMENSIONS attribute")

                # Check for standard_name
                if "standard_name" not in var.attrs:
                    issues.append("Missing standard_name attribute")

                # Check for grid_mapping (for data variables, not grid_mapping variables)
                if "grid_mapping" not in var.attrs and "grid_mapping_name" not in var.attrs:
                    issues.append("Missing grid_mapping attribute")

                if issues:
                    print(f"  ❌ {var_name}: {', '.join(issues)}")
                    compliance_issues.extend(issues)
                else:
                    print(f"  ✅ {var_name}: Compliant")
                    compliant_variables += 1

        print("\nSummary:")
        print("========")
        print(f"Total variables checked: {total_variables}")
        print(f"Compliant variables: {compliant_variables}")
        print(f"Non-compliant variables: {total_variables - compliant_variables}")

        if compliance_issues:
            print("\n❌ Dataset is NOT GeoZarr compliant")
            print(f"Issues found: {len(compliance_issues)}")
            if args.verbose:
                print("Detailed issues:")
                for issue in set(compliance_issues):
                    print(f"  - {issue}")
        else:
            print("\n✅ Dataset appears to be GeoZarr compliant")

    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for the CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="eopf-geozarr", description="Convert EOPF datasets to GeoZarr compliant format"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert EOPF dataset to GeoZarr compliant format"
    )
    convert_parser.add_argument(
        "input_path", type=str, help="Path to input EOPF dataset (Zarr format)"
    )
    convert_parser.add_argument("output_path", type=str, help="Path for output GeoZarr dataset (local path or S3 URL like s3://bucket/path)")
    convert_parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
        help="Groups to convert (default: Sentinel-2 resolution groups)",
    )
    convert_parser.add_argument(
        "--spatial-chunk",
        type=int,
        default=4096,
        help="Spatial chunk size for encoding (default: 4096)",
    )
    convert_parser.add_argument(
        "--min-dimension",
        type=int,
        default=256,
        help="Minimum dimension for overview levels (default: 256)",
    )
    convert_parser.add_argument(
        "--tile-width",
        type=int,
        default=256,
        help="Tile width for TMS compatibility (default: 256)",
    )
    convert_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for network operations (default: 3)",
    )
    convert_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    convert_parser.set_defaults(func=convert_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Display information about an EOPF dataset")
    info_parser.add_argument("input_path", type=str, help="Path to EOPF dataset")
    info_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    info_parser.set_defaults(func=info_command)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate GeoZarr compliance of a dataset"
    )
    validate_parser.add_argument("input_path", type=str, help="Path to dataset to validate")
    validate_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    validate_parser.set_defaults(func=validate_command)

    return parser


def main() -> None:
    """Execute main entry point for the CLI."""
    parser = create_parser()

    if len(sys.argv) == 1:
        # Show help if no arguments provided
        parser.print_help()
        return

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
