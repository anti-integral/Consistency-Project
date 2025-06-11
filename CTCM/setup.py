"""Setup script for Neural Operator Continuous Time Consistency Model."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-consistency-model",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Operator-based Continuous Time Consistency Models for fast high-quality generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/neural-consistency-model",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "einops>=0.6.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
        "Pillow>=9.0.0",
        "tensorboard>=2.11.0",
        "pytorch-fid>=0.3.0",
        "lpips>=0.1.4",
        "einops-exts>=0.0.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "noctcm-train=scripts.train_consistency:main",
            "noctcm-eval=scripts.evaluate:main",
            "noctcm-generate=scripts.generate_samples:main",
        ],
    },
)