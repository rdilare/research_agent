"""
Calculator Tool - Performs mathematical calculations and statistical analysis
"""
from typing import List, Dict, Any
import statistics
import re
import logging

logger = logging.getLogger(__name__)


class Calculator:
    """Tool for mathematical computations and statistical analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        number_pattern = r'-?\d+\.?\d*'
        numbers = []
        
        for match in re.finditer(number_pattern, text):
            try:
                num = float(match.group())
                # Filter out years and other non-meaningful numbers
                if not (1900 <= num <= 2100):  # Not likely a year
                    if 0.001 <= abs(num) <= 1000000:  # Reasonable range
                        numbers.append(num)
            except ValueError:
                continue
        
        return numbers
    
    def calculate_statistics(self, numbers: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of numbers"""
        if not numbers:
            return {}
        
        try:
            stats = {
                'count': len(numbers),
                'mean': statistics.mean(numbers),
                'median': statistics.median(numbers),
                'min': min(numbers),
                'max': max(numbers),
                'range': max(numbers) - min(numbers)
            }
            
            if len(numbers) > 1:
                stats['stdev'] = statistics.stdev(numbers)
                stats['variance'] = statistics.variance(numbers)
            
            return stats
        
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}
    
    def analyze_trends(self, numbers: List[float]) -> Dict[str, Any]:
        """Analyze trends in numerical data"""
        if len(numbers) < 2:
            return {}
        
        try:
            # Simple trend analysis
            differences = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            
            trend_analysis = {
                'data_points': numbers,
                'differences': differences,
                'average_change': statistics.mean(differences),
                'trend_direction': 'increasing' if statistics.mean(differences) > 0 else 'decreasing',
                'volatility': statistics.stdev(differences) if len(differences) > 1 else 0
            }
            
            return trend_analysis
        
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    def calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound annual growth rate"""
        if len(values) < 2 or values[0] == 0:
            return 0.0
        
        try:
            periods = len(values) - 1
            growth_rate = ((values[-1] / values[0]) ** (1/periods)) - 1
            return growth_rate * 100  # Return as percentage
        
        except Exception as e:
            logger.error(f"Growth rate calculation failed: {e}")
            return 0.0
    
    def calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation coefficient between two datasets"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            return statistics.correlation(x_values, y_values)
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0
    
    def financial_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate common financial ratios"""
        ratios = {}
        
        try:
            # Revenue-based ratios
            if 'revenue' in financial_data and 'costs' in financial_data:
                ratios['profit_margin'] = ((financial_data['revenue'] - financial_data['costs']) / financial_data['revenue']) * 100
            
            # Growth ratios
            if 'current_value' in financial_data and 'previous_value' in financial_data:
                if financial_data['previous_value'] != 0:
                    ratios['growth_rate'] = ((financial_data['current_value'] - financial_data['previous_value']) / financial_data['previous_value']) * 100
            
            return ratios
        
        except Exception as e:
            logger.error(f"Financial ratio calculation failed: {e}")
            return {}
