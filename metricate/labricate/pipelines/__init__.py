"""Pipeline implementations for Labricate.

Provides the default BERTopic-based pipeline and base protocol for custom pipelines.
"""

from metricate.labricate.pipelines.base import Pipeline
from metricate.labricate.pipelines.bertopic import BERTopicPipeline

__all__ = [
    "Pipeline",
    "BERTopicPipeline",
]
