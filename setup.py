import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agentpy", 
    version="0.0.3",
    author="Joël Foramitti",
    description="Agent-based modeling in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://agentpy.readthedocs.io/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    #setup_requires=['pytest-runner'],
    #tests_require=['pytest', 'pytest-cov'],
)