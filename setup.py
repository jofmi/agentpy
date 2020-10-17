import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agentpy", 
    version="0.0.2",
    author="Joël Foramitti",
    description="Agent-based modeling in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoelForamitti/agentpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)