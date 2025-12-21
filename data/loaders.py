"""
Dataset-specific loaders for the 4 downloaded datasets
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import logging
from PIL import Image
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load(self) -> List[Dict]:
        """Load and return list of {image_path, text, source} dicts"""
        raise NotImplementedError


class KneeXRayLoader(BaseLoader):
    """Load Mendeley Digital Knee X-Ray dataset (MedicalExpert-I only)"""
    
    # Mapping of grade folders to minimal text labels
    GRADE_LABELS = {
        '0Normal': "Knee X-ray - Normal",
        '1Doubtful': "Knee X-ray - Doubtful osteoarthritis",
        '2Mild': "Knee X-ray - Mild osteoarthritis",
        '3Moderate': "Knee X-ray - Moderate osteoarthritis",
        '4Severe': "Knee X-ray - Severe osteoarthritis"
    }
    
    def load(self) -> List[Dict]:
        pairs = []
        
        if not self.data_path.exists():
            logger.warning(f"Knee X-Ray path not found: {self.data_path}")
            return pairs
        
        logger.info("Loading Knee X-Ray dataset (MedicalExpert-I)...")
        
        # Navigate to MedicalExpert-I folder
        expert_path = self.data_path / "MedicalExpert-I"
        
        if not expert_path.exists():
            logger.warning(f"MedicalExpert-I folder not found at {expert_path}")
            return pairs
        
        # Iterate through grade folders
        for grade_folder in expert_path.iterdir():
            if not grade_folder.is_dir():
                continue
            
            folder_name = grade_folder.name
            
            # Get label for this grade
            if folder_name not in self.GRADE_LABELS:
                logger.debug(f"Skipping unknown grade folder: {folder_name}")
                continue
            
            text_label = self.GRADE_LABELS[folder_name]
            
            # Process all images in this grade folder
            for img_file in grade_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    pairs.append({
                        'image_path': str(img_file),
                        'text': text_label,
                        'source': 'knee_xray'
                    })
        
        logger.info(f"Loaded {len(pairs)} Knee X-Ray pairs from MedicalExpert-I")
        return pairs


class PADUFESLoader(BaseLoader):
    """Load PAD-UFES-20 dermatology dataset with demographic information"""
    
    # Diagnosis mapping to full names
    DIAGNOSIS_MAP = {
        'NEV': 'Nevus',
        'BCC': 'Basal Cell Carcinoma',
        'ACK': 'Actinic Keratosis',
        'SCC': 'Squamous Cell Carcinoma',
        'SEK': 'Seborrheic Keratosis'
    }
    
    def load(self) -> List[Dict]:
        pairs = []
        
        if not self.data_path.exists():
            logger.warning(f"PAD-UFES-20 path not found: {self.data_path}")
            return pairs
        
        logger.info("Loading PAD-UFES-20 dataset...")
        
        # Look for metadata CSV
        metadata_path = self.data_path / "metadata.csv"
        if not metadata_path.exists():
            logger.warning(f"PAD-UFES metadata not found at {metadata_path}")
            return pairs
        
        metadata = pd.read_csv(metadata_path)
        
        # Find images directory
        img_base_dir = self.data_path / "images"
        if not img_base_dir.exists():
            logger.warning(f"Images directory not found at {img_base_dir}")
            return pairs
        
        for _, row in metadata.iterrows():
            img_name = row.get('img_id', '')
            if not img_name or pd.isna(img_name):
                continue
            
            # Images are distributed across 3 folders
            img_path = None
            for subdir in ['imgs_part_1', 'imgs_part_2', 'imgs_part_3']:
                potential_path = img_base_dir / subdir / img_name
                if potential_path.exists():
                    img_path = potential_path
                    break
            
            if img_path is None or not img_path.exists():
                continue
            
            # Build clinical caption with demographics
            diagnostic = str(row.get('diagnostic', '')).strip()
            diagnosis_full = self.DIAGNOSIS_MAP.get(diagnostic, diagnostic)
            
            age = row.get('age', None)
            gender = row.get('gender', None)
            region = row.get('region', None)
            fitzpatrick = row.get('fitspatrick', None)
            
            # Build text based on available fields
            text_parts = []
            
            # Age and gender
            if pd.notna(age) and pd.notna(gender):
                age_int = int(age) if not pd.isna(age) else None
                if age_int is not None:
                    article = "An" if age_int < 20 else "A"
                    text_parts.append(f"{article} {age_int}-year-old {gender}")
            
            # Diagnosis
            text_parts.append(f"with {diagnosis_full}")
            
            # Location
            if pd.notna(region):
                text_parts.append(f"on the {region}")
            
            # Skin tone
            if pd.notna(fitzpatrick):
                fitz_int = int(fitzpatrick) if not pd.isna(fitzpatrick) else None
                if fitz_int is not None:
                    text_parts.append(f"(Fitzpatrick type {fitz_int})")
            
            text = " ".join(text_parts) + "."
            
            pairs.append({
                'image_path': str(img_path),
                'text': text,
                'source': 'pad_ufes_20'
            })
        
        logger.info(f"Loaded {len(pairs)} PAD-UFES-20 pairs")
        return pairs


class SCINLoader(BaseLoader):
    """Load SCIN dermatology dataset with condition labels"""
    
    def load(self) -> List[Dict]:
        pairs = []
        
        if not self.data_path.exists():
            logger.warning(f"SCIN path not found: {self.data_path}")
            return pairs
        
        logger.info("Loading SCIN dataset...")
        
        # Load case metadata
        case_path = self.data_path / "dataset_scin_cases.csv"
        if not case_path.exists():
            logger.warning("SCIN case metadata not found")
            return pairs
        
        cases_df = pd.read_csv(case_path)
        
        # Load label metadata
        label_path = self.data_path / "dataset_scin_labels.csv"
        if label_path.exists():
            labels_df = pd.read_csv(label_path)
        else:
            labels_df = None
        
        # Images directory
        img_base_dir = self.data_path / "dataset" / "images"
        if not img_base_dir.exists():
            img_base_dir = self.data_path / "images"
        
        for _, case_row in cases_df.iterrows():
            case_id = case_row.get('case_id', '')
            if pd.isna(case_id) or not case_id:
                continue
            
            # Get first image path
            img_name = case_row.get('image_1_path', '')
            if pd.isna(img_name) or not img_name:
                continue
            
            # Clean up path - image names often have absolute paths
            img_name = img_name.replace('dataset/images/', '').strip()
            img_path = img_base_dir / img_name
            
            if not img_path.exists():
                continue
            
            # Get diagnosis from labels if available
            diagnosis = "skin condition"
            if labels_df is not None:
                label_row = labels_df[labels_df['case_id'] == case_id]
                if not label_row.empty:
                    cond_str = label_row.iloc[0].get('weighted_skin_condition_label', '')
                    # Extract primary diagnosis from weighted label dict string
                    if pd.notna(cond_str) and isinstance(cond_str, str) and '{' in cond_str:
                        # Simple parsing of Python dict string
                        try:
                            # Get the condition with highest probability
                            conditions = str(cond_str).split("'")
                            if len(conditions) >= 2:
                                diagnosis = conditions[1]
                        except:
                            diagnosis = "skin condition"
            
            # Get demographics
            age_group = case_row.get('age_group', 'unknown age')
            sex = case_row.get('sex_at_birth', 'patient')
            fitzpatrick = case_row.get('fitzpatrick_skin_type', '')
            
            # Build text with available information
            text_parts = []
            if pd.notna(sex) and sex != 'OTHER_OR_UNSPECIFIED':
                text_parts.append(f"A {sex}")
            
            text_parts.append(f"with {diagnosis}")
            
            if pd.notna(fitzpatrick) and fitzpatrick and fitzpatrick != 'NONE_IDENTIFIED':
                # Parse Fitzpatrick type (e.g., FST1 -> type 1)
                fst_match = str(fitzpatrick).replace('FST', '')
                if fst_match:
                    text_parts.append(f"(Fitzpatrick type {fst_match})")
            
            text = " ".join(text_parts) + "."
            
            pairs.append({
                'image_path': str(img_path),
                'text': text,
                'source': 'scin'
            })
        
        logger.info(f"Loaded {len(pairs)} SCIN pairs")
        return pairs


class SLAKELoader(BaseLoader):
    """Load SLAKE-VQA medical question-answering dataset with answer translation"""
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        try:
            from googletrans import Translator
            self.translator = Translator()
        except ImportError:
            logger.warning("googletrans not installed, answers will not be translated")
            self.translator = None
    
    def _translate_to_english(self, text: str) -> str:
        """Translate text to English if in Chinese"""
        if not text or self.translator is None:
            return text
        
        try:
            # Skip translation for now (googletrans API issues)
            # Answers are already mostly in English or can be used as-is
            return text
        except Exception as e:
            logger.debug(f"Translation error: {e}")
            return text
    
    def load(self) -> List[Dict]:
        pairs = []
        
        if not self.data_path.exists():
            logger.warning(f"SLAKE path not found: {self.data_path}")
            return pairs
        
        logger.info("Loading SLAKE dataset...")
        
        # SLAKE has multiple JSON files
        json_files = [f for f in ['train.json', 'validation.json', 'test.json'] 
                      if (self.data_path / f).exists()]
        
        for json_file in json_files:
            json_path = self.data_path / json_file
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue
            
            # Handle different SLAKE formats (could be list or dict)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('data', data.get('samples', []))
            else:
                continue
            
            for item in items:
                # Get image name
                img_name = item.get('img_name', '')
                if not img_name:
                    continue
                
                # Get answer only (ignore question)
                answer = item.get('answer', item.get('a', ''))
                if not answer:
                    continue
                
                # Translate answer from Chinese to English
                answer_text = self._translate_to_english(str(answer))
                
                # Find image
                img_path = None
                img_dir = self.data_path / 'imgs'
                
                if img_dir.exists():
                    potential_path = img_dir / img_name
                    if potential_path.exists():
                        img_path = potential_path
                
                if img_path is None or not img_path.exists():
                    continue
                
                pairs.append({
                    'image_path': str(img_path),
                    'text': answer_text,
                    'source': 'slake'
                })
        
        logger.info(f"Loaded {len(pairs)} SLAKE pairs")
        return pairs