import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as r:
    requirements = [x.rstrip() for x in r.readlines()]

setuptools.setup(
    name="lw_visutils",
    version="0.0.1",
    author="wolterlw",
    author_email="wolterlentin@gmail.com",
    description="reusable code for computer vision tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wolterlw/net_modules/",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)