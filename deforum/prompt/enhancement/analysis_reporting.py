"""
Analysis and Reporting for Prompt Enhancement System

Statistics, reporting, and analysis functions following
functional programming principles with immutable data.
"""

from typing import Dict, Any, List
from .data_models import PromptEnhancementResult


def get_enhancement_statistics(result: PromptEnhancementResult) -> Dict[str, Any]:
    """Pure function: enhancement result -> statistics"""
    if not result.enhanced_prompts:
        return {
            "total_prompts": 0,
            "enhanced_count": 0,
            "success_rate": 0.0,
            "average_length_increase": 0.0,
            "processing_time": result.processing_time,
            "model_used": result.model_used
        }
    
    original_prompts = result.original_prompts
    enhanced_prompts = result.enhanced_prompts
    
    # Calculate length statistics using functional approach
    length_increases = tuple(
        len(enhanced_prompts.get(key, "")) - len(original_prompts.get(key, ""))
        for key in original_prompts.keys()
        if key in enhanced_prompts
    )
    
    average_length_increase = sum(length_increases) / len(length_increases) if length_increases else 0.0
    success_rate = result.enhancement_count / len(original_prompts) if original_prompts else 0.0
    
    return {
        "total_prompts": len(original_prompts),
        "enhanced_count": result.enhancement_count,
        "success_rate": success_rate,
        "average_length_increase": average_length_increase,
        "processing_time": result.processing_time,
        "model_used": result.model_used,
        "language": result.language.value,
        "success": result.success
    }


def format_enhancement_report(result: PromptEnhancementResult) -> str:
    """Pure function: enhancement result -> formatted report"""
    stats = get_enhancement_statistics(result)
    
    if not result.success:
        return f"""❌ Prompt Enhancement Failed
Error: {result.error_message}
Model: {result.model_used}
Processing Time: {result.processing_time:.2f}s"""
    
    return f"""✅ Prompt Enhancement Complete!

📊 Statistics:
- Total Prompts: {stats['total_prompts']}
- Enhanced: {stats['enhanced_count']}
- Success Rate: {stats['success_rate']:.1%}
- Average Length Increase: +{stats['average_length_increase']:.0f} characters
- Model Used: {stats['model_used']}
- Language: {stats['language'].title()}
- Processing Time: {stats['processing_time']:.2f}s

The enhanced prompts are ready for video generation!"""


def calculate_processing_metrics(result: PromptEnhancementResult) -> Dict[str, float]:
    """Pure function: enhancement result -> processing performance metrics"""
    stats = get_enhancement_statistics(result)
    
    if stats['total_prompts'] == 0:
        return {
            "prompts_per_second": 0.0,
            "average_time_per_prompt": 0.0,
            "characters_processed_per_second": 0.0,
            "enhancement_efficiency": 0.0
        }
    
    prompts_per_second = stats['total_prompts'] / max(result.processing_time, 0.001)
    average_time_per_prompt = result.processing_time / stats['total_prompts']
    
    # Calculate total characters processed
    total_original_chars = sum(len(prompt) for prompt in result.original_prompts.values())
    characters_per_second = total_original_chars / max(result.processing_time, 0.001)
    
    # Enhancement efficiency (success rate weighted by processing speed)
    enhancement_efficiency = stats['success_rate'] * prompts_per_second
    
    return {
        "prompts_per_second": prompts_per_second,
        "average_time_per_prompt": average_time_per_prompt,
        "characters_processed_per_second": characters_per_second,
        "enhancement_efficiency": enhancement_efficiency
    }


def analyze_enhancement_quality(result: PromptEnhancementResult) -> Dict[str, Any]:
    """Pure function: enhancement result -> quality analysis"""
    if not result.enhanced_prompts or not result.original_prompts:
        return {
            "average_quality_score": 0.0,
            "quality_distribution": {},
            "top_improvements": [],
            "improvement_summary": {}
        }
    
    # Calculate quality scores for each prompt
    quality_scores = []
    improvements = []
    
    for key in result.enhanced_prompts:
        if key in result.original_prompts:
            original = result.original_prompts[key]
            enhanced = result.enhanced_prompts[key]
            
            # Simple quality scoring
            length_improvement = len(enhanced) - len(original)
            quality_score = min(length_improvement / max(len(original), 1), 2.0)  # Cap at 200% improvement
            
            quality_scores.append(quality_score)
            improvements.append({
                "frame": key,
                "original_length": len(original),
                "enhanced_length": len(enhanced),
                "improvement": length_improvement,
                "quality_score": quality_score
            })
    
    # Calculate statistics
    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    # Distribution analysis
    quality_ranges = {
        "excellent": sum(1 for score in quality_scores if score >= 1.5),
        "good": sum(1 for score in quality_scores if 1.0 <= score < 1.5),
        "moderate": sum(1 for score in quality_scores if 0.5 <= score < 1.0),
        "minimal": sum(1 for score in quality_scores if score < 0.5)
    }
    
    # Top improvements
    top_improvements = sorted(improvements, key=lambda x: x['quality_score'], reverse=True)[:5]
    
    # Summary statistics
    improvement_summary = {
        "total_original_chars": sum(len(prompt) for prompt in result.original_prompts.values()),
        "total_enhanced_chars": sum(len(prompt) for prompt in result.enhanced_prompts.values()),
        "average_length_increase": sum(imp['improvement'] for imp in improvements) / len(improvements) if improvements else 0,
        "max_improvement": max(imp['improvement'] for imp in improvements) if improvements else 0,
        "min_improvement": min(imp['improvement'] for imp in improvements) if improvements else 0
    }
    
    return {
        "average_quality_score": average_quality,
        "quality_distribution": quality_ranges,
        "top_improvements": top_improvements,
        "improvement_summary": improvement_summary
    }


def generate_detailed_report(result: PromptEnhancementResult) -> str:
    """Pure function: enhancement result -> detailed analysis report"""
    basic_stats = get_enhancement_statistics(result)
    performance_metrics = calculate_processing_metrics(result)
    quality_analysis = analyze_enhancement_quality(result)
    
    if not result.success:
        return f"""
❌ PROMPT ENHANCEMENT FAILED

Error Details:
- Error Message: {result.error_message}
- Model Attempted: {result.model_used}
- Processing Time: {result.processing_time:.2f}s
- Language: {result.language.value}

Please check your model configuration and try again.
"""
    
    report = f"""
✅ PROMPT ENHANCEMENT DETAILED REPORT

📈 BASIC STATISTICS:
- Total Prompts: {basic_stats['total_prompts']}
- Successfully Enhanced: {basic_stats['enhanced_count']}
- Success Rate: {basic_stats['success_rate']:.1%}
- Model Used: {basic_stats['model_used']}
- Language: {basic_stats['language'].title()}

⚡ PERFORMANCE METRICS:
- Processing Time: {result.processing_time:.2f}s
- Prompts per Second: {performance_metrics['prompts_per_second']:.1f}
- Average Time per Prompt: {performance_metrics['average_time_per_prompt']:.2f}s
- Characters per Second: {performance_metrics['characters_processed_per_second']:.0f}
- Enhancement Efficiency: {performance_metrics['enhancement_efficiency']:.2f}

🎯 QUALITY ANALYSIS:
- Average Quality Score: {quality_analysis['average_quality_score']:.2f}
- Quality Distribution:
  • Excellent (≥1.5): {quality_analysis['quality_distribution']['excellent']} prompts
  • Good (1.0-1.5): {quality_analysis['quality_distribution']['good']} prompts
  • Moderate (0.5-1.0): {quality_analysis['quality_distribution']['moderate']} prompts
  • Minimal (<0.5): {quality_analysis['quality_distribution']['minimal']} prompts

📊 IMPROVEMENT SUMMARY:
- Total Characters: {quality_analysis['improvement_summary']['total_original_chars']} → {quality_analysis['improvement_summary']['total_enhanced_chars']}
- Average Length Increase: +{quality_analysis['improvement_summary']['average_length_increase']:.0f} characters
- Maximum Improvement: +{quality_analysis['improvement_summary']['max_improvement']} characters
- Minimum Improvement: +{quality_analysis['improvement_summary']['min_improvement']} characters
"""
    
    # Add top improvements if available
    if quality_analysis['top_improvements']:
        report += "\n🏆 TOP IMPROVEMENTS:\n"
        for i, imp in enumerate(quality_analysis['top_improvements'][:3], 1):
            report += f"  {i}. Frame {imp['frame']}: {imp['original_length']} → {imp['enhanced_length']} chars (+{imp['improvement']}, score: {imp['quality_score']:.2f})\n"
    
    report += "\nThe enhanced prompts are ready for video generation! 🚀"
    
    return report


def compare_enhancement_results(results: List[PromptEnhancementResult]) -> Dict[str, Any]:
    """Pure function: multiple enhancement results -> comparison analysis"""
    if not results:
        return {"error": "No results to compare"}
    
    if len(results) < 2:
        return {"error": "Need at least 2 results for comparison"}
    
    comparison = {
        "total_results": len(results),
        "models_compared": [result.model_used for result in results],
        "best_performance": {},
        "best_quality": {},
        "summary": {}
    }
    
    # Compare performance metrics
    performance_data = [calculate_processing_metrics(result) for result in results]
    
    # Find best performing result for each metric
    best_speed_idx = max(range(len(performance_data)), 
                        key=lambda i: performance_data[i]['prompts_per_second'])
    best_efficiency_idx = max(range(len(performance_data)), 
                             key=lambda i: performance_data[i]['enhancement_efficiency'])
    
    comparison["best_performance"] = {
        "fastest_model": results[best_speed_idx].model_used,
        "fastest_speed": performance_data[best_speed_idx]['prompts_per_second'],
        "most_efficient_model": results[best_efficiency_idx].model_used,
        "highest_efficiency": performance_data[best_efficiency_idx]['enhancement_efficiency']
    }
    
    # Compare quality metrics
    quality_data = [analyze_enhancement_quality(result) for result in results]
    best_quality_idx = max(range(len(quality_data)), 
                          key=lambda i: quality_data[i]['average_quality_score'])
    
    comparison["best_quality"] = {
        "highest_quality_model": results[best_quality_idx].model_used,
        "quality_score": quality_data[best_quality_idx]['average_quality_score']
    }
    
    # Overall summary
    success_rates = [get_enhancement_statistics(result)['success_rate'] for result in results]
    best_success_idx = max(range(len(success_rates)), key=lambda i: success_rates[i])
    
    comparison["summary"] = {
        "most_reliable_model": results[best_success_idx].model_used,
        "highest_success_rate": success_rates[best_success_idx],
        "average_success_rate": sum(success_rates) / len(success_rates),
        "total_prompts_processed": sum(len(result.original_prompts) for result in results)
    }
    
    return comparison


def export_results_to_dict(result: PromptEnhancementResult) -> Dict[str, Any]:
    """Pure function: enhancement result -> exportable dictionary"""
    return {
        "metadata": {
            "model_used": result.model_used,
            "language": result.language.value,
            "processing_time": result.processing_time,
            "success": result.success,
            "error_message": result.error_message,
            "enhancement_count": result.enhancement_count
        },
        "statistics": get_enhancement_statistics(result),
        "performance_metrics": calculate_processing_metrics(result),
        "quality_analysis": analyze_enhancement_quality(result),
        "prompts": {
            "original": result.original_prompts,
            "enhanced": result.enhanced_prompts
        }
    }
