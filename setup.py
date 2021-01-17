import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agentpy",
    license="BSD 3-Clause",
    author="Joël Foramitti",
    author_email="joel.foramitti@uab.cat",
    description="Agent-based modeling in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://agentpy.readthedocs.io/",
    download_url="https://github.com/JoelForamitti/agentpy",
    install_requires=[
        "numpy >= 1.19"
        "matplotlib >= 3.3.3"
        "networkx >= 2.5"
        "pandas >= 1.1.3"
        "SALib >= 1.3.7"
        "IPython >= 7.15.0"
        "ipywidgets >= 7.5.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent"
    ],
)
