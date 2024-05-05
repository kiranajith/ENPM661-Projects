from setuptools import setup
import os
from glob import glob

package_name = 'project5_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name),
         glob('launch/*launch.[pxy][yma]*')),
        # (os.path.join('share', package_name, 'config/'),
        #  glob('config/*')),
        (os.path.join('share', package_name, 'maps/'),
         glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kiran',
    maintainer_email='kiranajith97@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'amcl_publisher = project5_pkg.set_init_amcl_pose:main',
            'a_star = project5_pkg.a_star:main',
        ],
    },
)