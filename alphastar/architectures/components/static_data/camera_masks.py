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

"""List of function arguments which are forbidden outside the camera view."""

import numpy as np
from pysc2.lib import actions as sc2_actions


def get_on_camera_only_functions_unit() -> np.ndarray:
  """Get numpy arrays for forbidden function aruments (for target_unit_tag).

  A function is allowed if this function can target a unit outside the camera
  view.

  Returns:
    A numpy array or size [num_functions] with allowed actions.
  """
  # The comments represent the proportion of this action being taken inside the
  # camera view. A ratio smaller than 1 means that it is possible to take the
  # action outside the camera view, but a ratio too very close to 1 can
  # probably be attributed to extremely rare cases of lag.
  forbidden_strings_unit = [
      "Effect_Heal_unit",  # 1.0
      "Load_unit",  # 1.0
      "UnloadAllAt_unit",  # 0.999983450056
      "Effect_Repair_unit",  # 0.999948423984
      "Effect_ParasiticBomb_unit",  # 0.999918923301
      "Effect_NeuralParasite_unit",  # 0.999915218313
      "Effect_InterferenceMatrix_unit",  # 0.999906153954
      "Effect_Transfusion_unit",  # 0.999903032766
      "Effect_KD8Charge_unit",  # 0.999860876478
      "Effect_GravitonBeam_unit",  # 0.999811122204
      "Effect_Feedback_unit",  # 0.999803439803
      "Effect_AntiArmorMissile_unit",  # 0.999800478851
      "Rally_Workers_unit",  # 0.99979222557
      "Harvest_Gather_unit",  # 0.999783643444
      "Effect_GhostSnipe_unit",  # 0.999712542302
      "Effect_CalldownMULE_unit",  # 0.999684635984
      "Effect_YamatoGun_unit",  # 0.999666009288
      "Effect_Abduct_unit",  # 0.99961916632
      "Effect_Restore_unit",  # 0.999610894942
      "Effect_LockOn_unit",  # 0.999435436919
      "Effect_CausticSpray_unit",  # 0.999386466788
      "Build_Assimilator_unit",  # 0.999380867711
      "Effect_ChronoBoostEnergyCost_unit",  # 0.999369350264
      "Effect_SupplyDrop_unit",  # 0.999364486992
      "Attack_unit",  # 0.999226272993
      "Build_Extractor_unit",  # 0.999015262677
  ]

  num_functions = len(sc2_actions.RAW_FUNCTIONS)
  func_forbidden_unit = np.zeros((num_functions,), dtype=np.bool_)
  for func in sc2_actions.RAW_FUNCTIONS:
    if func.name in forbidden_strings_unit:
      func_forbidden_unit[func.id] = True

  return func_forbidden_unit


def get_on_camera_only_functions_pt() -> np.ndarray:
  """Get numpy arrays for forbidden function aruments (for world targets).

  A function is allowed if this function can target a point outside the camera
  view.

  Returns:
    A numpy array or size [num_functions] with allowed actions.
  """
  # See get_on_camera_only_functions_unit for an explanation about this:
  forbidden_strings_pt = [
      "Build_SpawningPool_pt",  # 0.999961021547
      "Build_RoboticsFacility_pt",  # 0.999951444054
      "Build_DarkShrine_pt",  # 0.999946532642
      "Build_ShieldBattery_pt",  # 0.999942637826
      "Build_CyberneticsCore_pt",  # 0.999935187586
      "Build_FleetBeacon_pt",  # 0.999916742986
      "Build_Forge_pt",  # 0.999885483468
      "Build_Bunker_pt",  # 0.999880587034
      "Build_TwilightCouncil_pt",  # 0.999878848251
      "TrainWarp_Sentry_pt",  # 0.999874913631
      "Build_RoboticsBay_pt",  # 0.999865885824
      "Build_EvolutionChamber_pt",  # 0.999857662698
      "Build_Gateway_pt",  # 0.99983839885
      "Build_RoachWarren_pt",  # 0.999834649776
      "Build_LurkerDen_pt",  # 0.999834011121
      "Build_Reactor_pt",  # 0.999822511059
      "Build_PhotonCannon_pt",  # 0.999820207885
      "Build_TemplarArchive_pt",  # 0.999805560957
      "Build_Factory_pt",  # 0.999803283379
      "Build_UltraliskCavern_pt",  # 0.999794175157
      "Build_Stargate_pt",  # 0.999792180443
      "Effect_KD8Charge_pt",  # 0.999764604339
      "Build_BanelingNest_pt",  # 0.999760468917
      "Effect_ForceField_pt",  # 0.999744805733
      "Effect_BlindingCloud_pt",  # 0.999743754004
      "Build_Barracks_pt",  # 0.999720537569
      "Build_GhostAcademy_pt",  # 0.99971667375
      "Build_InfestationPit_pt",  # 0.999707345625
      "Build_Starport_pt",  # 0.999704161829
      "TrainWarp_Adept_pt",  # 0.999697424477
      "Build_SpineCrawler_pt",  # 0.999697112121
      "Build_NydusNetwork_pt",  # 0.999696251747
      "TrainWarp_HighTemplar_pt",  # 0.999682031856
      "TrainWarp_DarkTemplar_pt",  # 0.999670937893
      "Build_HydraliskDen_pt",  # 0.999667068958
      "Effect_PsiStorm_pt",  # 0.999665857415
      "Build_Nexus_pt",  # 0.999633286184
      "Build_Hatchery_pt",  # 0.999602838197
      "Build_TechLab_pt",  # 0.999594232302
      "Build_EngineeringBay_pt",  # 0.999573728563
      "Morph_Root_pt",  # 0.999563520376
      "Build_NydusWorm_pt",  # 0.99955992372
      "Build_Armory_pt",  # 0.99951750906
      "Build_SporeCrawler_pt",  # 0.999503242441
      "Effect_EMP_pt",  # 0.999490282118
      "Build_Spire_pt",  # 0.999481813652
      "Effect_FungalGrowth_pt",  # 0.999471675961
      "Build_SupplyDepot_pt",  # 0.999392261968
      "Effect_CorrosiveBile_pt",  # 0.999334492724
      "Build_FusionCore_pt",  # 0.999280989359
      "TrainWarp_Zealot_pt",  # 0.999219225426
      "TrainWarp_Stalker_pt",  # 0.999179110584
      "Build_Pylon_pt",  # 0.999056181889
      "Effect_TimeWarp_pt",  # 0.999025341131
      "Build_CommandCenter_pt",  # 0.998844091799
      "Build_MissileTurret_pt",  # 0.998724833923
      "Land_pt",  # 0.998663265556
      "Effect_InfestedTerrans_pt",  # 0.998277927523
      "Build_SensorTower_pt",  # 0.998016675332
      "Build_Refinery_pt",  # 0.997900839664
      "Build_StasisTrap_pt",  # 0.997851289226
      "Effect_OracleRevelation_pt",  # 0.997267759563
      "Effect_AutoTurret_pt",  # 0.997062686567
      "Effect_PurificationNova_pt",  # 0.995978149949
  ]
  num_functions = len(sc2_actions.RAW_FUNCTIONS)
  func_forbidden_pt = np.zeros((num_functions,), dtype=np.bool_)
  for func in sc2_actions.RAW_FUNCTIONS:
    if func.name in forbidden_strings_pt:
      func_forbidden_pt[func.id] = 1

  return func_forbidden_pt
