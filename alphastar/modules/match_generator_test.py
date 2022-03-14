# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for match_generator.py."""

from pysc2.env import sc2_env
from alphastar.modules import match_generator
from absl.testing import absltest


class MatchGeneratorTest(absltest.TestCase):

  def test_error_cases(self):
    with self.assertRaisesRegex(ValueError, 'must be non-empty'):
      match_generator.MatchGenerator([], [sc2_env.Race.terran], ['Acropolis'])
    with self.assertRaisesRegex(ValueError, 'must be non-empty'):
      match_generator.MatchGenerator([sc2_env.Race.terran], None, ['Acropolis'])
    with self.assertRaisesRegex(ValueError, 'must be non-empty'):
      match_generator.MatchGenerator([sc2_env.Race.zerg], [sc2_env.Race.terran],
                                     [])

  def test_build_in_bot(self):
    gen = match_generator.MatchGenerator([sc2_env.Race.zerg],
                                         [sc2_env.Race.zerg], ['Acropolis'])
    match = gen.generate('very_easy')
    self.assertEqual(
        match,
        match_generator.Match(sc2_env.Race.zerg, sc2_env.Race.zerg,
                              'Acropolis'))


if __name__ == '__main__':
  absltest.main()
