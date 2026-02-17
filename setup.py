#!/usr/bin/env python3
"""
Setup configuration for LLM Benchmark Reporter
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README_PACKAGE.md").read_text() if (this_directory / "README_PACKAGE.md").exists() else ""

setup(
    name="llm-benchmark-reporter",
    version="1.0.0",
    author="Your Organization",
    author_email="benchmarking@yourorg.com",
    description="Professional benchmarking and reporting tool for LLM endpoints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/llm-benchmark-reporter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "reportlab>=4.0.0",
        "pillow>=9.0.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-benchmark-report=llm_benchmark_reporter.cli:main",
            "llm-benchmark-stats=llm_benchmark_reporter.stats:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_benchmark_reporter": ["templates/*", "examples/*"],
    },
    zip_safe=False,
)
