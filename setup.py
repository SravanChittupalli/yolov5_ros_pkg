from setuptools import find_packages, setup

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sravanchittupalli',
    maintainer_email='sravanchittupalli7@gmail.com',
    description='Pallet Detection andGround segmentation using yolov5',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_image = yolov5_ros.publish_image:main',
            'predict = yolov5_ros.predict:main'
        ],
    },
)
