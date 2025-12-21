"""
Script to prepare unified medical dataset from 4 downloaded datasets
"""
import argparse
import pandas as pd
from pathlib import Path
import logging
from config import Config
from data.loaders import (
    KneeXRayLoader,
    PADUFESLoader,
    SLAKELoader
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_datasets(config):
    """Load all 4 available datasets"""
    all_pairs = []
    
    # Knee X-Ray
    logger.info("Loading Knee X-Ray...")
    try:
        loader = KneeXRayLoader(config.data.knee_xray_path)
        pairs = loader.load()
        all_pairs.extend(pairs)
    except Exception as e:
        logger.warning(f"Could not load Knee X-Ray: {e}")
    
    # PAD-UFES-20
    logger.info("Loading PAD-UFES-20...")
    try:
        loader = PADUFESLoader(config.data.pad_ufes_path)
        pairs = loader.load()
        all_pairs.extend(pairs)
    except Exception as e:
        logger.warning(f"Could not load PAD-UFES-20: {e}")
    
    # SLAKE
    logger.info("Loading SLAKE...")
    try:
        loader = SLAKELoader(config.data.slake_path)
        pairs = loader.load()
        all_pairs.extend(pairs)
    except Exception as e:
        logger.warning(f"Could not load SLAKE: {e}")
    
    return all_pairs


def apply_dataset_weights(df, weights):
    """Apply sampling weights to balance datasets"""
    weighted_dfs = []
    
    for source, weight in weights.items():
        source_df = df[df['source'] == source]
        if len(source_df) > 0:
            # Calculate number of samples to keep
            n_samples = int(len(source_df) * weight)
            if n_samples > 0:
                sampled = source_df.sample(n=min(n_samples, len(source_df)), random_state=42)
                weighted_dfs.append(sampled)
                logger.info(f"{source}: {len(source_df)} -> {len(sampled)} samples (weight={weight})")
    
    return pd.concat(weighted_dfs, ignore_index=True)


def clean_dataset(df):
    """Clean and validate dataset"""
    initial_count = len(df)
    
    # Remove rows with missing values
    df = df.dropna()
    logger.info(f"Removed {initial_count - len(df)} rows with missing values")
    
    # Remove rows with very short text (< 10 characters)
    df = df[df['text'].str.len() >= 10]
    logger.info(f"Removed rows with text < 10 characters")
    
    # Check if image files exist
    def image_exists(path):
        return Path(path).exists()
    
    df['exists'] = df['image_path'].apply(image_exists)
    missing = (~df['exists']).sum()
    if missing > 0:
        logger.warning(f"Found {missing} missing image files")
    df = df[df['exists']].drop('exists', axis=1)
    
    logger.info(f"Final dataset size: {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare unified medical dataset from 4 datasets")
    parser.add_argument('--no-weights', action='store_true', help='Skip dataset weighting')
    parser.add_argument('--output', type=str, help='Output parquet path')
    
    args = parser.parse_args()
    config = Config()
    
    # Load all datasets
    logger.info("="*60)
    logger.info("Loading 4 medical datasets...")
    logger.info("="*60)
    
    all_pairs = load_all_datasets(config)
    
    if len(all_pairs) == 0:
        logger.error("No data loaded! Please check dataset paths in config.py")
        logger.error("Current paths:")
        logger.error(f"  Knee X-Ray: {config.data.knee_xray_path}")
        logger.error(f"  PAD-UFES-20: {config.data.pad_ufes_path}")
        logger.error(f"  SCIN: {config.data.scin_path}")
        logger.error(f"  SLAKE: {config.data.slake_path}")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_pairs)
    logger.info(f"\nTotal pairs loaded: {len(df)}")
    logger.info(f"Sources: {df['source'].unique()}")
    logger.info("\nDataset distribution:")
    logger.info(df['source'].value_counts())
    
    # Apply dataset weights (optional)
    if not args.no_weights:
        logger.info("\n" + "="*60)
        logger.info("Applying dataset weights...")
        logger.info("="*60)
        df = apply_dataset_weights(df, config.data.dataset_weights)
        logger.info(f"After weighting: {len(df)} pairs")
    
    # Clean dataset
    logger.info("\n" + "="*60)
    logger.info("Cleaning dataset...")
    logger.info("="*60)
    df = clean_dataset(df)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Dataset shuffled")
    
    # Train/val split
    train_df, val_df = train_test_split(
        df,
        test_size=config.data.val_split,
        random_state=42,
        stratify=df['source']
    )
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    final_df = pd.concat([train_df, val_df], ignore_index=True)
    
    logger.info(f"\nTrain samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    
    # Save to parquet
    output_path = args.output or config.data.processed_data_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    final_df.to_parquet(output_path, index=False)
    logger.info(f"\n" + "="*60)
    logger.info(f"Saved unified dataset to: {output_path}")
    logger.info("="*60)
    
    # Save statistics
    stats = {
        'total_samples': len(final_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'sources': df['source'].value_counts().to_dict(),
        'train_sources': train_df['source'].value_counts().to_dict(),
        'val_sources': val_df['source'].value_counts().to_dict()
    }
    
    stats_path = Path(output_path).parent / "dataset_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved statistics to: {stats_path}")
    logger.info("\nNext step: python train.py")


if __name__ == "__main__":
    main()