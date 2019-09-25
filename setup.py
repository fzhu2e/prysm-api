from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='prysm-api',
    version='0.5.1',
    description="The API for PRoxY System Modeling (PRYSM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author='Sylvia Dee, Amir Allam, and Feng Zhu',
    url='https://github.com/fzhu2e/prysm-api',
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='prysm-api',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'fbm',
        'rpy2',
        'tqdm',
        'pathos',
    ],
)
