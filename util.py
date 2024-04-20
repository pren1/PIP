import pdb
import re
from collections import namedtuple

# Define namedtuple structures with additional fields for the UDP address, sensor data, and timestamp
SensorData = namedtuple('SensorData', ['type', 'timestamp', 'udp_address', 'sensor_id', 'data'])

def parse_sensor_data(data_str):
    # Regex to capture the timestamp, UDP address, and sensor ID
    # prefix_regex = r"Time:\[(\d+)\],\[udp://([0-9A-Fa-f:]+)/(\d+)\]"
    prefix_regex = r"\[udp://([0-9A-Fa-f:]+)/(\d+)\]"
    prefix_match = re.search(prefix_regex, data_str)
    if not prefix_match:
        return None  # If the prefix doesn't match, return None

    # timestamp = int(prefix_match.group(1))
    timestamp = -1
    udp_address = prefix_match.group(1)
    sensor_id = prefix_match.group(2)

    # Regex for Quaternion and Vector3
    # quaternion_regex = r"Quaternion\(w=([-\d\.]+), x=([-\d\.]+), y=([-\d\.]+), z=([-\d\.]+)\)"
    # vector3_regex = r"Vector3\(x=([-\d\.]+), y=([-\d\.]+), z=([-\d\.]+)\)"
    quaternion_regex = r"Quaternion\(w=(-?\d+\.?\d*(?:E[-+]?\d+)?), x=(-?\d+\.?\d*(?:E[-+]?\d+)?), y=(-?\d+\.?\d*(?:E[-+]?\d+)?), z=(-?\d+\.?\d*(?:E[-+]?\d+)?)\)"
    vector3_regex = r"Vector3\(x=(-?\d+\.?\d*(?:E[-+]?\d+)?), y=(-?\d+\.?\d*(?:E[-+]?\d+)?), z=(-?\d+\.?\d*(?:E[-+]?\d+)?)\)"

    # pdb.set_trace()
    # Try to match Quaternion
    quaternion_match = re.search(quaternion_regex, data_str)
    if quaternion_match:
        quaternion = (float(quaternion_match.group(1)), float(quaternion_match.group(2)),
                      float(quaternion_match.group(3)), float(quaternion_match.group(4)))
        return SensorData(type='Rotation', timestamp=timestamp, udp_address=udp_address, sensor_id=sensor_id, data=quaternion)

    # Try to match Vector3
    vector3_match = re.search(vector3_regex, data_str)
    if vector3_match:
        vector3 = (float(vector3_match.group(1)), float(vector3_match.group(2)), float(vector3_match.group(3)))
        return SensorData(type='Acceleration', timestamp=timestamp, udp_address=udp_address, sensor_id=sensor_id, data=vector3)

    return None

if __name__ == '__main__':
    test_str = 'Time:[1713441841620],[udp://58:BF:25:D1:03:A9/0] Acceleration value: Vector3(x=-0.026544893, y=-3.1233561E-4, z=0.15774433)\n'
    # test_str = 'Time:[1713442311141],[udp://58:BF:25:D1:03:A9/0] Acceleration value: Vector3(x=-0.04399258, y=-0.0014832792, z=0.14534312)\n'
    print(parse_sensor_data(test_str))