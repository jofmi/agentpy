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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent"
    ],
)
