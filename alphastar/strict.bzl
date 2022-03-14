# Copyright 2021 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BUILD rules for compatibility with strict targets."""

def py_strict_binary(name, **kwargs):
    native.py_binary(name = name, **kwargs)

def py_strict_test(name, **kwargs):
    native.py_test(name = name, **kwargs)

def py_library_with_strictdeps_info(name, **kwargs):
    native.py_library(name = name, **kwargs)
