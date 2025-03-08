from setuptools import setup, find_packages

setup(
    name="footwear-impression-matching",
    version="0.1.0",
    description="Deep learning system for matching crime scene footwear impressions to reference databases",
    author="JJshome",
    author_email="",
    url="https://github.com/JJshome/footwear-impression-matching",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
