from glob import glob
from setuptools import find_packages, setup


package_name = "franka_lerobot_rviz"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}", ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pnp",
    maintainer_email="pnp@example.com",
    description="Read-only RViz visualization for LeRobot Cartesian plans and Franka execution.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "franka_action_chunk_visualizer = "
            "franka_lerobot_rviz.action_chunk_visualizer:main",
        ],
    },
)
