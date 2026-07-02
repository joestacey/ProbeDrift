from setuptools import setup, find_packages

setup(
    name='probe_drift',
    version='0.1.0',
    packages=find_packages(),
    package_data={'probe_drift': ['benchmark_splits/*.json.gz']},
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.8',
)
