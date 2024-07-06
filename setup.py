from setuptools import setup, find_packages

setup(
    name="spectral-bridges",
    version="0.1.5",
    author="Félix Laplante",
    author_email="flheight0@gmail.com",
    description="Spectral Bridges clustering algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flheight/spectral-bridges-pypi/",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "faiss-cpu",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
    ],
    python_requires='>=3.6',
)
