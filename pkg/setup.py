import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepmil-trela", # Replace with your own username
    version="0.0.1",
    author="Trite Zard",
    author_email="trislaz@hotmail.fr",
    description="Multiple instance learning for wsi classification.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trislaz/deepMIL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
