import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agentpy", 
    version="0.0.1",
    author="Joël Foramitti",
    description="A framework for the development and analysis of agent-based models with multiple agent types and environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoelForamitti/agentpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)