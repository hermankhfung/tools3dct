import setuptools

setuptools.setup(
    name='tools3dct',
    version='0.0.10',
    author='Herman Fung',
    url='https://git.embl.de/fung/tools3dct',
    packages=setuptools.find_packages(),
    scripts=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
    install_requires=[
        'tifffile>=2020.2.16',
        'scikit-image>=0.14.1',
        'numpy>=1.15.4',
        'scipy>=1.1.0',
        'PyQt5>=5.12.2',
        'psutil>=3.3.0'
    ],
    extras_require={
        'opencv-python': ['opencv-python>=4.0.1'],
        'opencv-python-headless': ['opencv-python-headless>=4.0.1']
    }
)