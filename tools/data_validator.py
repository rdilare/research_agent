"""
Data Validator Tool - Validates data quality, completeness, and consistency
"""
from typing import Dict, Any, List
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class DataValidator:
    """Tool for validating research data quality and relevance"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = {
            'completeness': self._check_completeness,
            'format': self._check_format,
            'consistency': self._check_consistency,
            'relevance': self._check_relevance,
            'quality': self._check_quality
        }
    
    def validate_context_data(self, context: List[str]) -> Dict[str, Any]:
        """
        Validate the quality and relevance of context data
        
        Args:
            context: List of text passages from research
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'total_passages': len(context),
                'validation_timestamp': datetime.now().isoformat(),
                'overall_score': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            if not context:
                validation_results['issues'].append('No context data provided')
                validation_results['overall_score'] = 0.0
                return validation_results
            
            # Run validation checks
            scores = []
            
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    rule_result = rule_func(context)
                    validation_results[rule_name] = rule_result
                    scores.append(rule_result.get('score', 0.0))
                    
                    # Collect issues and recommendations
                    if rule_result.get('issues'):
                        validation_results['issues'].extend(rule_result['issues'])
                    if rule_result.get('recommendations'):
                        validation_results['recommendations'].extend(rule_result['recommendations'])
                        
                except Exception as e:
                    logger.warning(f"Validation rule {rule_name} failed: {e}")
                    validation_results['issues'].append(f"Validation rule {rule_name} failed: {e}")
            
            # Calculate overall score
            validation_results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(f"Data validation completed. Overall score: {validation_results['overall_score']:.2f}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'total_passages': len(context) if context else 0
            }
    
    def _check_completeness(self, context: List[str]) -> Dict[str, Any]:
        """Check data completeness"""
        result = {
            'score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check for minimum content
        total_length = sum(len(text) for text in context)
        avg_length = total_length / len(context) if context else 0
        
        if total_length < 500:
            result['issues'].append('Insufficient content volume (less than 500 characters)')
            result['recommendations'].append('Gather more comprehensive data sources')
            result['score'] = 0.3
        elif avg_length < 50:
            result['issues'].append('Very short passages detected')
            result['recommendations'].append('Look for more detailed sources')
            result['score'] = 0.6
        else:
            result['score'] = 1.0
        
        # Check for empty or near-empty passages
        empty_count = sum(1 for text in context if len(text.strip()) < 20)
        if empty_count > 0:
            result['issues'].append(f'{empty_count} passages are too short or empty')
            result['score'] *= 0.8
        
        return result
    
    def _check_format(self, context: List[str]) -> Dict[str, Any]:
        """Check data format consistency"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check for encoding issues
        encoding_issues = 0
        for text in context:
            # Look for common encoding problems
            if '\\x' in text or '\\u' in text:
                encoding_issues += 1
        
        if encoding_issues > 0:
            result['issues'].append(f'{encoding_issues} passages have encoding issues')
            result['recommendations'].append('Check source data encoding and cleanup')
            result['score'] *= 0.7
        
        # Check for excessive HTML/markup
        html_issues = 0
        html_pattern = re.compile(r'<[^>]+>')
        for text in context:
            if len(html_pattern.findall(text)) > 5:
                html_issues += 1
        
        if html_issues > 0:
            result['issues'].append(f'{html_issues} passages contain excessive HTML markup')
            result['recommendations'].append('Improve HTML cleanup in data processing')
            result['score'] *= 0.8
        
        return result
    
    def _check_consistency(self, context: List[str]) -> Dict[str, Any]:
        """Check data consistency across passages"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check for contradictory information (simplified)
        # This is a basic implementation - could be enhanced with NLP
        
        # Look for numerical inconsistencies
        numbers = []
        for text in context:
            found_numbers = re.findall(r'-?\d+\.?\d*', text)
            numbers.extend([float(n) for n in found_numbers if self._is_valid_number(n)])
        
        if numbers:
            # Check for extreme outliers
            if len(numbers) > 3:
                numbers_sorted = sorted(numbers)
                q1 = numbers_sorted[len(numbers)//4]
                q3 = numbers_sorted[3*len(numbers)//4]
                iqr = q3 - q1
                
                outliers = [n for n in numbers if n < q1 - 1.5*iqr or n > q3 + 1.5*iqr]
                if len(outliers) > len(numbers) * 0.2:  # More than 20% outliers
                    result['issues'].append('High number of statistical outliers detected')
                    result['recommendations'].append('Review data sources for accuracy')
                    result['score'] *= 0.8
        
        return result
    
    def _check_relevance(self, context: List[str]) -> Dict[str, Any]:
        """Check relevance of data to research context"""
        result = {
            'score': 0.8,  # Default assumption of relevance
            'issues': [],
            'recommendations': []
        }
        
        # Check for very generic or boilerplate content
        generic_patterns = [
            r'copyright \d{4}',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'cookie policy'
        ]
        
        generic_count = 0
        for text in context:
            for pattern in generic_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    generic_count += 1
                    break
        
        if generic_count > len(context) * 0.3:  # More than 30% generic
            result['issues'].append('High amount of generic/boilerplate content')
            result['recommendations'].append('Filter out non-content elements during scraping')
            result['score'] *= 0.6
        
        # Check for minimum content diversity
        unique_starts = set()
        for text in context:
            if len(text) > 50:
                unique_starts.add(text[:50].lower())
        
        diversity_ratio = len(unique_starts) / len(context) if context else 0
        if diversity_ratio < 0.5:
            result['issues'].append('Low content diversity detected')
            result['recommendations'].append('Expand data sources for more diverse perspectives')
            result['score'] *= 0.7
        
        return result
    
    def _check_quality(self, context: List[str]) -> Dict[str, Any]:
        """Check overall data quality indicators"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check for readability and structure
        poorly_structured = 0
        for text in context:
            # Basic checks for poor structure
            sentences = text.split('.')
            if len(sentences) > 1:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if avg_sentence_length > 50 or avg_sentence_length < 3:
                    poorly_structured += 1
        
        if poorly_structured > len(context) * 0.2:
            result['issues'].append('Poor text structure detected in multiple passages')
            result['recommendations'].append('Improve content extraction and formatting')
            result['score'] *= 0.8
        
        # Check for language consistency
        non_english_count = 0
        for text in context:
            # Simple heuristic for non-English content
            ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text) if text else 1
            if ascii_ratio < 0.8:  # Less than 80% ASCII characters
                non_english_count += 1
        
        if non_english_count > len(context) * 0.1:  # More than 10% non-English
            result['issues'].append('Mixed language content detected')
            result['recommendations'].append('Apply language filtering to maintain consistency')
            result['score'] *= 0.9
        
        return result
    
    def _is_valid_number(self, num_str: str) -> bool:
        """Check if a string represents a valid number for analysis"""
        try:
            num = float(num_str)
            # Filter out years and other non-analytical numbers
            return not (1800 <= num <= 2100) and abs(num) < 1e10
        except ValueError:
            return False
    
    def validate_data_structure(self, data: Any, expected_schema: Dict[str, Any]) -> bool:
        """Validate data against expected schema"""
        try:
            if isinstance(expected_schema, dict):
                if not isinstance(data, dict):
                    return False
                
                for key, expected_type in expected_schema.items():
                    if key not in data:
                        return False
                    if not isinstance(data[key], expected_type):
                        return False
                
                return True
            
            return isinstance(data, expected_schema)
            
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
