# Copyright 2026 pnp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import yaml


def test_observation_camera_displays_match_client_inputs():
    config_path = Path(__file__).parents[1] / 'rviz' / 'action_viz.rviz'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    displays = config['Visualization Manager']['Displays']
    image_displays = {
        display['Name']: display
        for display in displays
        if display.get('Class') == 'rviz_default_plugins/Image'
    }

    expected = {
        'Observation camera1 (base_0_rgb)': '/camera1/camera1/color/image_raw',
        'Observation camera2 (left_wrist_0_rgb)': '/camera2/camera2/color/image_raw',
    }
    assert set(image_displays) == set(expected)
    for name, topic in expected.items():
        display = image_displays[name]
        assert display['Enabled'] is True
        assert display['Topic']['Value'] == topic
        assert display['Topic']['Depth'] == 1
        assert display['Topic']['Reliability Policy'] == 'Best Effort'
        assert display['Topic']['Durability Policy'] == 'Volatile'
