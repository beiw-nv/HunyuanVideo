#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Functions for loading models.
"""
from __future__ import annotations

import gc
import glob
import os
import sys
from typing import List, Optional

import onnx
import torch


def onnx_graph_needs_external_data(onnx_graph: onnx.ModelProto) -> bool:
    """Return true if ONNX graph needs to store external data."""
    if sys.platform == "win32":
        # ByteSize is broken (wraps around) on Windows, so always assume external data is needed.
        return True
    else:
        TWO_GIGABYTES = 2147483648
        return onnx_graph.ByteSize() > TWO_GIGABYTES


