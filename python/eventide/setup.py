from setuptools import setup, find_packages

setup(
    name="eventide",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        # include any shared‐object (.so) files in the eventide package
        "eventide": ["_eventide*.so"],
    },
    include_package_data=True,
    zip_safe=False,
)
