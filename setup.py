#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
with open('README.md') as readme_file:
    readme = readme_file.read()
requirements = [
    "tensorflow",
    "numpy",
    "matplotlib",
    "seaborn",
    "jupyter",
    "scikit-learn==0.23.2",
    #"brain-score @ git+https://github.com/brain-score/brain-score.git@v1.3",
    #"result_caching @ git+https://github.com/brain-score/result_caching",
    "netCDF4",
    "pillow",
    "llist",
    "nbsvm",
    "tqdm",
    "transformers"
    ]

setup(
    name='ann_brain_alignment',
    version='0.1.0',
    description="",
    long_description=readme,
    python_requires='>=3.6, <4',
    author="Eghbal Hosseini",
    author_email='ehoseini@mit.edu',
    url='https://github.com/eghbalhosseini/ann_brain_alignment',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='computational neuroscience, human language, '
             'machine learning, deep neural networks',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.9',
    ],
)
