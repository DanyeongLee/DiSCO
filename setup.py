from setuptools import setup, find_packages

setup(
    name='disco',
    version='0.1.0', 
    packages=find_packages(),
    include_package_data=True,
    description='Official code implementation for DiSCO in PyTorch (Lee et al., AAAI 2024)',
    author='Danyeong Lee',
    author_email='ldy9381@snu,ac,kr',
    url='https://github.com/DanyeongLee/DiSCO',  # URL to the repo
    license='MIT',  # Choose an appropriate license
    classifiers=[
        'Programming Language :: Python :: 3',  # Specify Python versions supported
        'License :: OSI Approved :: MIT License',  # License information
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version requirement
    install_requires=[],  # No additional packages required
)
