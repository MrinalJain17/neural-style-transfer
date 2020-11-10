from setuptools import setup, find_packages

setup(
    name="neural_style_transfer",
    version="0.0.0",
    description="Neural Stye Transfer",
    author="Mrinal Jain",
    author_email="mrinal.jain@nyu.edu",
    url="https://github.com/MrinalJain17/neural-style-transfer",
    install_requires=[],
    packages=find_packages(exclude=["images"]),
)
