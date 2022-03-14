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

"""Lookups for the drastic torso preprocessors."""

import enum
import itertools
from typing import List, Mapping, Optional

import chex
import numpy as np
from pysc2.env.converter.cc.game_data.python import uint8_lookup
from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units
from pysc2.lib.features import FeatureUnit

from s2clientprotocol import raw_pb2 as sc_raw


INITIAL_MINERALS_CONTENTS = {
    sc2_units.Neutral.RichMineralField: 1500,
    sc2_units.Neutral.RichMineralField750: 750,
    sc2_units.Neutral.MineralField: 1800,
    sc2_units.Neutral.MineralField750: 750,
    sc2_units.Neutral.LabMineralField: 1800,
    sc2_units.Neutral.LabMineralField750: 750,
    sc2_units.Neutral.PurifierRichMineralField: 1500,
    sc2_units.Neutral.PurifierRichMineralField750: 750,
    sc2_units.Neutral.BattleStationMineralField: 1500,
    sc2_units.Neutral.BattleStationMineralField750: 750,
    sc2_units.Neutral.PurifierMineralField: 1500,
    sc2_units.Neutral.PurifierMineralField750: 750,
}
INITIAL_VESPENE_CONTENTS = {
    sc2_units.Neutral.VespeneGeyser: 2250,
    sc2_units.Neutral.RichVespeneGeyser: 2500,
    sc2_units.Neutral.PurifierVespeneGeyser: 2250,
    sc2_units.Neutral.ShakurasVespeneGeyser: 2250,
}
INITIAL_RESOURCE_CONTENTS = dict(itertools.chain(
    INITIAL_MINERALS_CONTENTS.items(), INITIAL_VESPENE_CONTENTS.items()))


MAX_VALUES = {
    FeatureUnit.alliance: max(sc_raw.Alliance.values()),
    FeatureUnit.health: 1500,
    FeatureUnit.shield: 1000,
    FeatureUnit.energy: 200,
    FeatureUnit.cargo_space_taken: 8,
    FeatureUnit.display_type: max(sc_raw.DisplayType.values()),
    FeatureUnit.cloak: max(sc_raw.CloakState.values()),
    FeatureUnit.is_powered: 1,
    FeatureUnit.mineral_contents: max(INITIAL_MINERALS_CONTENTS.values()),
    FeatureUnit.vespene_contents: max(INITIAL_VESPENE_CONTENTS.values()),
    FeatureUnit.cargo_space_max: 8,
    FeatureUnit.assigned_harvesters: 24,
    FeatureUnit.ideal_harvesters: 16,
    FeatureUnit.weapon_cooldown: 32,
    FeatureUnit.order_length: 8,
    FeatureUnit.hallucination: 1,
    FeatureUnit.active: 1,
    FeatureUnit.is_on_screen: 1,
    FeatureUnit.is_blip: 1,
    FeatureUnit.order_progress_0: 100,
    FeatureUnit.order_progress_1: 100,
    FeatureUnit.is_in_cargo: 1,
    FeatureUnit.buff_duration_remain: 250,
    FeatureUnit.attack_upgrade_level: 4,
    FeatureUnit.armor_upgrade_level: 4,
    FeatureUnit.shield_upgrade_level: 4,
    # Previous arguments:
    -2: 1,
    -1: 1,
}


# This table maps the redundant and unused order id to their remapped version.
# Everything not in this map is not remapped, so this is future-proof.
REDUNDANT_GENERIC_ORDER_ID = {
    "Attack_Attack_pt":
        "Attack_pt",
    "Attack_AttackBuilding_pt":
        "Attack_pt",
    "Attack_Attack_unit":
        "Attack_unit",
    "Attack_AttackBuilding_unit":
        "Attack_unit",
    "Attack_Battlecruiser_pt":
        "Attack_pt",
    "Attack_Battlecruiser_unit":
        "Attack_unit",
    "Attack_Redirect_pt":
        "Attack_pt",
    "Attack_Redirect_unit":
        "Attack_unit",
    "Behavior_BuildingAttackOff_quick":
        "no_op",
    "Behavior_BuildingAttackOn_quick":
        "no_op",
    "Behavior_CloakOff_Banshee_quick":
        "Behavior_CloakOff_quick",
    "Behavior_CloakOff_Ghost_quick":
        "Behavior_CloakOff_quick",
    "Behavior_CloakOn_Banshee_quick":
        "Behavior_CloakOn_quick",
    "Behavior_CloakOn_Ghost_quick":
        "Behavior_CloakOn_quick",
    "Behavior_HoldFireOff_Ghost_quick":
        "Behavior_HoldFireOff_quick",
    "Behavior_HoldFireOff_Lurker_quick":
        "Behavior_HoldFireOff_quick",
    "Behavior_HoldFireOn_Ghost_quick":
        "Behavior_HoldFireOn_quick",
    "Behavior_HoldFireOn_Lurker_quick":
        "Behavior_HoldFireOn_quick",
    "Build_CreepTumor_Queen_pt":
        "Build_CreepTumor_pt",
    "Build_CreepTumor_Tumor_pt":
        "Build_CreepTumor_pt",
    "Build_Reactor_Barracks_pt":
        "Build_Reactor_pt",
    "Build_Reactor_Barracks_quick":
        "Build_Reactor_quick",
    "Build_Reactor_Factory_pt":
        "Build_Reactor_pt",
    "Build_Reactor_Factory_quick":
        "Build_Reactor_quick",
    "Build_Reactor_Starport_pt":
        "Build_Reactor_pt",
    "Build_Reactor_Starport_quick":
        "Build_Reactor_quick",
    "Build_TechLab_Barracks_pt":
        "Build_TechLab_pt",
    "Build_TechLab_Barracks_quick":
        "Build_TechLab_quick",
    "Build_TechLab_Factory_pt":
        "Build_TechLab_pt",
    "Build_TechLab_Factory_quick":
        "Build_TechLab_quick",
    "Build_TechLab_Starport_pt":
        "Build_TechLab_pt",
    "Build_TechLab_Starport_quick":
        "Build_TechLab_quick",
    "BurrowDown_Baneling_quick":
        "BurrowDown_quick",
    "BurrowDown_Drone_quick":
        "BurrowDown_quick",
    "BurrowDown_Hydralisk_quick":
        "BurrowDown_quick",
    "BurrowDown_Infestor_quick":
        "BurrowDown_quick",
    "BurrowDown_InfestorTerran_quick":
        "BurrowDown_quick",
    "BurrowDown_Lurker_quick":
        "BurrowDown_quick",
    "BurrowDown_Queen_quick":
        "BurrowDown_quick",
    "BurrowDown_Ravager_quick":
        "BurrowDown_quick",
    "BurrowDown_Roach_quick":
        "BurrowDown_quick",
    "BurrowDown_SwarmHost_quick":
        "BurrowDown_quick",
    "BurrowDown_Ultralisk_quick":
        "BurrowDown_quick",
    "BurrowDown_WidowMine_quick":
        "BurrowDown_quick",
    "BurrowDown_Zergling_quick":
        "BurrowDown_quick",
    "BurrowUp_Baneling_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Baneling_quick":
        "BurrowUp_quick",
    "BurrowUp_Drone_quick":
        "BurrowUp_quick",
    "BurrowUp_Hydralisk_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Hydralisk_quick":
        "BurrowUp_quick",
    "BurrowUp_Infestor_quick":
        "BurrowUp_quick",
    "BurrowUp_InfestorTerran_autocast":
        "BurrowUp_autocast",
    "BurrowUp_InfestorTerran_quick":
        "BurrowUp_quick",
    "BurrowUp_Lurker_quick":
        "BurrowUp_quick",
    "BurrowUp_Queen_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Queen_quick":
        "BurrowUp_quick",
    "BurrowUp_Ravager_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Ravager_quick":
        "BurrowUp_quick",
    "BurrowUp_Roach_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Roach_quick":
        "BurrowUp_quick",
    "BurrowUp_SwarmHost_quick":
        "BurrowUp_quick",
    "BurrowUp_Ultralisk_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Ultralisk_quick":
        "BurrowUp_quick",
    "BurrowUp_WidowMine_quick":
        "BurrowUp_quick",
    "BurrowUp_Zergling_autocast":
        "BurrowUp_autocast",
    "BurrowUp_Zergling_quick":
        "BurrowUp_quick",
    "Cancel_AdeptPhaseShift_quick":
        "Cancel_quick",
    "Cancel_AdeptShadePhaseShift_quick":
        "Cancel_quick",
    "Cancel_BarracksADDON_quick":
        "Cancel_quick",
    "Cancel_BuildInProgress_quick":
        "Cancel_quick",
    "Cancel_CreepTumor_quick":
        "Cancel_quick",
    "Cancel_FactoryADDON_quick":
        "Cancel_quick",
    "Cancel_GravitonBeam_quick":
        "Cancel_quick",
    "Cancel_HangarQueue5_quick":
        "Cancel_quick",
    "Cancel_Last_quick":
        "Cancel_quick",
    "Cancel_LockOn_quick":
        "Cancel_quick",
    "Cancel_MorphBroodlord_quick":
        "Cancel_quick",
    "Cancel_MorphGreaterSpire_quick":
        "Cancel_quick",
    "Cancel_MorphHive_quick":
        "Cancel_quick",
    "Cancel_MorphLair_quick":
        "Cancel_quick",
    "Cancel_MorphLurkerDen_quick":
        "Cancel_quick",
    "Cancel_MorphLurker_quick":
        "Cancel_quick",
    "Cancel_MorphMothership_quick":
        "Cancel_quick",
    "Cancel_MorphOrbital_quick":
        "Cancel_quick",
    "Cancel_MorphOverlordTransport_quick":
        "Cancel_quick",
    "Cancel_MorphOverseer_quick":
        "Cancel_quick",
    "Cancel_MorphPlanetaryFortress_quick":
        "Cancel_quick",
    "Cancel_MorphRavager_quick":
        "Cancel_quick",
    "Cancel_MorphThorExplosiveMode_quick":
        "Cancel_quick",
    "Cancel_NeuralParasite_quick":
        "Cancel_quick",
    "Cancel_Nuke_quick":
        "Cancel_quick",
    "Cancel_Queue1_quick":
        "Cancel_quick",
    "Cancel_Queue5_quick":
        "Cancel_quick",
    "Cancel_QueueADDON_quick":
        "Cancel_quick",
    "Cancel_QueueCancelToSelection_quick":
        "Cancel_quick",
    "Cancel_QueuePassiveCancelToSelection_quick":
        "Cancel_quick",
    "Cancel_QueuePassive_quick":
        "Cancel_quick",
    "Cancel_SpineCrawlerRoot_quick":
        "Cancel_quick",
    "Cancel_SporeCrawlerRoot_quick":
        "Cancel_quick",
    "Cancel_StarportADDON_quick":
        "Cancel_quick",
    "Cancel_StasisTrap_quick":
        "Cancel_quick",
    "Cancel_VoidRayPrismaticAlignment_quick":
        "Cancel_quick",
    "Effect_Blink_Stalker_pt":
        "Effect_Blink_pt",
    "Effect_ChronoBoost_unit":
        "Effect_ChronoBoostEnergyCost_unit",
    "Effect_MassRecall_Mothership_pt":
        "Effect_MassRecall_pt",
    "Effect_MassRecall_Nexus_pt":
        "Effect_MassRecall_pt",
    "Effect_MassRecall_StrategicRecall_pt":
        "Effect_MassRecall_pt",
    "Effect_Repair_Mule_autocast":
        "Effect_Repair_autocast",
    "Effect_Repair_Mule_unit":
        "Effect_Repair_unit",
    "Effect_Repair_RepairDrone_autocast":
        "Effect_Repair_autocast",
    "Effect_Repair_RepairDrone_unit":
        "Effect_Repair_unit",
    "Effect_Repair_SCV_autocast":
        "Effect_Repair_autocast",
    "Effect_Repair_SCV_unit":
        "Effect_Repair_unit",
    "Effect_ShadowStride_pt":
        "Effect_Blink_pt",
    "Effect_Spray_pt":
        "no_op",
    "Effect_Spray_Protoss_pt":
        "no_op",
    "Effect_Spray_Terran_pt":
        "no_op",
    "Effect_Spray_Zerg_pt":
        "no_op",
    "Effect_Stim_Marauder_quick":
        "Effect_Stim_quick",
    "Effect_Stim_Marauder_Redirect_quick":
        "Effect_Stim_quick",
    "Effect_Stim_Marine_quick":
        "Effect_Stim_quick",
    "Effect_Stim_Marine_Redirect_quick":
        "Effect_Stim_quick",
    "Effect_WidowMineAttack_pt":
        "Attack_pt",
    "Effect_WidowMineAttack_unit":
        "Attack_unit",
    "Halt_Building_quick":
        "Halt_quick",
    "Halt_TerranBuild_quick":
        "Halt_quick",
    "Harvest_Gather_Drone_pt":
        "Harvest_Gather_unit",
    "Harvest_Gather_Mule_pt":
        "Harvest_Gather_unit",
    "Harvest_Gather_Probe_pt":
        "Harvest_Gather_unit",
    "Harvest_Gather_SCV_pt":
        "Harvest_Gather_unit",
    "Harvest_Gather_Drone_unit":
        "Harvest_Gather_unit",
    "Harvest_Gather_Mule_unit":
        "Harvest_Gather_unit",
    "Harvest_Gather_Probe_unit":
        "Harvest_Gather_unit",
    "Harvest_Gather_SCV_unit":
        "Harvest_Gather_unit",
    "Harvest_Return_Drone_quick":
        "Harvest_Return_quick",
    "Harvest_Return_Mule_quick":
        "Harvest_Return_quick",
    "Harvest_Return_Probe_quick":
        "Harvest_Return_quick",
    "Harvest_Return_SCV_quick":
        "Harvest_Return_quick",
    "HoldPosition_Battlecruiser_quick":
        "HoldPosition_quick",
    "HoldPosition_Hold_quick":
        "HoldPosition_quick",
    "Land_Barracks_pt":
        "Land_pt",
    "Land_CommandCenter_pt":
        "Land_pt",
    "Land_Factory_pt":
        "Land_pt",
    "Land_OrbitalCommand_pt":
        "Land_pt",
    "Land_Starport_pt":
        "Land_pt",
    "Lift_Barracks_quick":
        "Lift_quick",
    "Lift_CommandCenter_quick":
        "Lift_quick",
    "Lift_Factory_quick":
        "Lift_quick",
    "Lift_OrbitalCommand_quick":
        "Lift_quick",
    "Lift_Starport_quick":
        "Lift_quick",
    "LoadAll_CommandCenter_quick":
        "LoadAll_quick",
    "Load_Bunker_unit":
        "Load_unit",
    "Load_Medivac_unit":
        "Load_unit",
    "Load_NydusNetwork_unit":
        "Load_unit",
    "Load_NydusWorm_unit":
        "Load_unit",
    "Load_Overlord_unit":
        "Load_unit",
    "Load_WarpPrism_unit":
        "Load_unit",
    "Morph_LurkerDen_quick":
        "Build_LurkerDen_pt",
    "Morph_Mothership_quick":
        "no_op",
    "Morph_SpineCrawlerRoot_pt":
        "Morph_Root_pt",
    "Morph_SpineCrawlerUproot_quick":
        "Morph_Uproot_quick",
    "Morph_SporeCrawlerRoot_pt":
        "Morph_Root_pt",
    "Morph_SporeCrawlerUproot_quick":
        "Morph_Uproot_quick",
    "Move_Battlecruiser_pt":
        "Move_pt",
    "Move_Battlecruiser_unit":
        "Move_unit",
    "Move_Move_pt":
        "Move_pt",
    "Move_Move_unit":
        "Move_unit",
    "Patrol_Battlecruiser_pt":
        "Patrol_pt",
    "Patrol_Battlecruiser_unit":
        "Patrol_unit",
    "Patrol_Patrol_pt":
        "Patrol_pt",
    "Patrol_Patrol_unit":
        "Patrol_unit",
    "Rally_CommandCenter_pt":
        "Rally_Building_pt",
    "Rally_CommandCenter_unit":
        "Rally_Building_unit",
    "Rally_Hatchery_Units_pt":
        "Rally_Building_pt",
    "Rally_Hatchery_Units_unit":
        "Rally_Building_unit",
    "Rally_Hatchery_Workers_pt":
        "Rally_Building_pt",
    "Rally_Hatchery_Workers_unit":
        "Rally_Building_unit",
    "Rally_Morphing_Unit_pt":
        "Rally_Building_pt",
    "Rally_Morphing_Unit_unit":
        "Rally_Building_unit",
    "Rally_Nexus_pt":
        "Rally_Building_pt",
    "Rally_Nexus_unit":
        "Rally_Building_unit",
    "Rally_Units_pt":
        "Rally_Building_pt",
    "Rally_Units_unit":
        "Rally_Building_unit",
    "Rally_Workers_pt":
        "Rally_Building_pt",
    "Rally_Workers_unit":
        "Rally_Building_unit",
    "Research_NeosteelFrame_quick":
        "Research_TerranStructureArmorUpgrade_quick",
    "Research_ProtossAirArmorLevel1_quick":
        "Research_ProtossAirArmor_quick",
    "Research_ProtossAirArmorLevel2_quick":
        "Research_ProtossAirArmor_quick",
    "Research_ProtossAirArmorLevel3_quick":
        "Research_ProtossAirArmor_quick",
    "Research_ProtossAirWeaponsLevel1_quick":
        "Research_ProtossAirWeapons_quick",
    "Research_ProtossAirWeaponsLevel2_quick":
        "Research_ProtossAirWeapons_quick",
    "Research_ProtossAirWeaponsLevel3_quick":
        "Research_ProtossAirWeapons_quick",
    "Research_ProtossGroundArmorLevel1_quick":
        "Research_ProtossGroundArmor_quick",
    "Research_ProtossGroundArmorLevel2_quick":
        "Research_ProtossGroundArmor_quick",
    "Research_ProtossGroundArmorLevel3_quick":
        "Research_ProtossGroundArmor_quick",
    "Research_ProtossGroundWeaponsLevel1_quick":
        "Research_ProtossGroundWeapons_quick",
    "Research_ProtossGroundWeaponsLevel2_quick":
        "Research_ProtossGroundWeapons_quick",
    "Research_ProtossGroundWeaponsLevel3_quick":
        "Research_ProtossGroundWeapons_quick",
    "Research_ProtossShieldsLevel1_quick":
        "Research_ProtossShields_quick",
    "Research_ProtossShieldsLevel2_quick":
        "Research_ProtossShields_quick",
    "Research_ProtossShieldsLevel3_quick":
        "Research_ProtossShields_quick",
    "Research_TerranInfantryArmorLevel1_quick":
        "Research_TerranInfantryArmor_quick",
    "Research_TerranInfantryArmorLevel2_quick":
        "Research_TerranInfantryArmor_quick",
    "Research_TerranInfantryArmorLevel3_quick":
        "Research_TerranInfantryArmor_quick",
    "Research_TerranInfantryWeaponsLevel1_quick":
        "Research_TerranInfantryWeapons_quick",
    "Research_TerranInfantryWeaponsLevel2_quick":
        "Research_TerranInfantryWeapons_quick",
    "Research_TerranInfantryWeaponsLevel3_quick":
        "Research_TerranInfantryWeapons_quick",
    "Research_TerranShipWeaponsLevel1_quick":
        "Research_TerranShipWeapons_quick",
    "Research_TerranShipWeaponsLevel2_quick":
        "Research_TerranShipWeapons_quick",
    "Research_TerranShipWeaponsLevel3_quick":
        "Research_TerranShipWeapons_quick",
    "Research_TerranVehicleAndShipPlatingLevel1_quick":
        "Research_TerranVehicleAndShipPlating_quick",
    "Research_TerranVehicleAndShipPlatingLevel2_quick":
        "Research_TerranVehicleAndShipPlating_quick",
    "Research_TerranVehicleAndShipPlatingLevel3_quick":
        "Research_TerranVehicleAndShipPlating_quick",
    "Research_TerranVehicleWeaponsLevel1_quick":
        "Research_TerranVehicleWeapons_quick",
    "Research_TerranVehicleWeaponsLevel2_quick":
        "Research_TerranVehicleWeapons_quick",
    "Research_TerranVehicleWeaponsLevel3_quick":
        "Research_TerranVehicleWeapons_quick",
    "Research_ZergFlyerArmorLevel1_quick":
        "Research_ZergFlyerArmor_quick",
    "Research_ZergFlyerArmorLevel2_quick":
        "Research_ZergFlyerArmor_quick",
    "Research_ZergFlyerArmorLevel3_quick":
        "Research_ZergFlyerArmor_quick",
    "Research_ZergFlyerAttackLevel1_quick":
        "Research_ZergFlyerAttack_quick",
    "Research_ZergFlyerAttackLevel2_quick":
        "Research_ZergFlyerAttack_quick",
    "Research_ZergFlyerAttackLevel3_quick":
        "Research_ZergFlyerAttack_quick",
    "Research_ZergGroundArmorLevel1_quick":
        "Research_ZergGroundArmor_quick",
    "Research_ZergGroundArmorLevel2_quick":
        "Research_ZergGroundArmor_quick",
    "Research_ZergGroundArmorLevel3_quick":
        "Research_ZergGroundArmor_quick",
    "Research_ZergMeleeWeaponsLevel1_quick":
        "Research_ZergMeleeWeapons_quick",
    "Research_ZergMeleeWeaponsLevel2_quick":
        "Research_ZergMeleeWeapons_quick",
    "Research_ZergMeleeWeaponsLevel3_quick":
        "Research_ZergMeleeWeapons_quick",
    "Research_ZergMissileWeaponsLevel1_quick":
        "Research_ZergMissileWeapons_quick",
    "Research_ZergMissileWeaponsLevel2_quick":
        "Research_ZergMissileWeapons_quick",
    "Research_ZergMissileWeaponsLevel3_quick":
        "Research_ZergMissileWeapons_quick",
    "Stop_Battlecruiser_quick":
        "Stop_quick",
    "Stop_Building_quick":
        "Stop_quick",
    "Stop_Redirect_quick":
        "Stop_quick",
    "Stop_Stop_quick":
        "Stop_quick",
    "Train_MothershipCore_quick":
        "no_op",
    "UnloadAllAt_Medivac_pt":
        "UnloadAllAt_pt",
    "UnloadAllAt_Medivac_unit":
        "UnloadAllAt_unit",
    "UnloadAllAt_Overlord_pt":
        "UnloadAllAt_pt",
    "UnloadAllAt_Overlord_unit":
        "UnloadAllAt_unit",
    "UnloadAllAt_WarpPrism_pt":
        "UnloadAllAt_pt",
    "UnloadAllAt_WarpPrism_unit":
        "UnloadAllAt_unit",
    "UnloadAll_Bunker_quick":
        "UnloadAll_quick",
    "UnloadAll_CommandCenter_quick":
        "UnloadAll_quick",
    "UnloadAll_NydusNetwork_quick":
        "UnloadAll_quick",
    "UnloadAll_NydusWorm_quick":
        "UnloadAll_quick",
}


ADDON_UNIT_TYPES = [
    sc2_units.Terran.BarracksTechLab,
    sc2_units.Terran.BarracksReactor,
    sc2_units.Terran.FactoryTechLab,
    sc2_units.Terran.FactoryReactor,
    sc2_units.Terran.StarportTechLab,
    sc2_units.Terran.StarportReactor,
]


class UA(enum.IntEnum):
  """Unit attributes."""
  LIGHT = 0
  ARMORED = 1
  BIOLOGICAL = 2
  MECHANICAL = 3
  PSIONIC = 4
  MASSIVE = 5
  STRUCTURE = 6
  DETECTOR = 7
  SUMMONED = 8
  FLYING = 9
  ADDON = 10
  BURROWED = 11


UNITS_ATTRIBUTES = {
    # Protoss
    sc2_units.Protoss.Adept: [UA.LIGHT, UA.BIOLOGICAL],
    sc2_units.Protoss.AdeptPhaseShift: [UA.LIGHT, UA.BIOLOGICAL, UA.SUMMONED],
    sc2_units.Protoss.Archon: [UA.PSIONIC, UA.MASSIVE],
    sc2_units.Protoss.Assimilator: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.Carrier: [
        UA.ARMORED, UA.MASSIVE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Protoss.Colossus: [UA.ARMORED, UA.MASSIVE, UA.MECHANICAL],
    sc2_units.Protoss.CyberneticsCore: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.DarkShrine: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.DarkTemplar: [UA.BIOLOGICAL, UA.LIGHT, UA.PSIONIC],
    sc2_units.Protoss.Disruptor: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Protoss.DisruptorPhased: [UA.SUMMONED],
    sc2_units.Protoss.FleetBeacon: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.ForceField: [UA.SUMMONED],
    sc2_units.Protoss.Forge: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.Gateway: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.HighTemplar: [UA.BIOLOGICAL, UA.LIGHT, UA.PSIONIC],
    sc2_units.Protoss.Immortal: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Protoss.Interceptor: [
        UA.LIGHT, UA.MECHANICAL, UA.SUMMONED, UA.FLYING
    ],
    sc2_units.Protoss.Mothership: [
        UA.ARMORED, UA.MASSIVE, UA.PSIONIC, UA.MECHANICAL, UA.FLYING
    ],
    # sc2_units.Protoss.MothershipCore: [UA.MECHANICAL, UA.ARMORED, UA.PSIONIC],
    sc2_units.Protoss.Nexus: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.Observer: [
        UA.LIGHT, UA.MECHANICAL, UA.DETECTOR, UA.FLYING
    ],
    sc2_units.Protoss.ObserverSurveillanceMode: [
        UA.LIGHT, UA.MECHANICAL, UA.DETECTOR, UA.FLYING
    ],
    sc2_units.Protoss.Oracle: [
        UA.MECHANICAL, UA.ARMORED, UA.PSIONIC, UA.FLYING
    ],
    sc2_units.Protoss.Phoenix: [UA.LIGHT, UA.MECHANICAL, UA.FLYING],
    sc2_units.Protoss.PhotonCannon: [UA.ARMORED, UA.STRUCTURE, UA.DETECTOR],
    sc2_units.Protoss.Probe: [UA.LIGHT, UA.MECHANICAL],
    sc2_units.Protoss.Pylon: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.PylonOvercharged: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.RoboticsBay: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.RoboticsFacility: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.Sentry: [UA.LIGHT, UA.MECHANICAL, UA.PSIONIC],
    sc2_units.Protoss.ShieldBattery: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.Stalker: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Protoss.Stargate: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.StasisTrap: [UA.LIGHT, UA.STRUCTURE, UA.SUMMONED],
    sc2_units.Protoss.Tempest: [
        UA.ARMORED, UA.MECHANICAL, UA.MASSIVE, UA.FLYING
    ],
    sc2_units.Protoss.TemplarArchive: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.TwilightCouncil: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.VoidRay: [UA.ARMORED, UA.MECHANICAL, UA.FLYING],
    sc2_units.Protoss.WarpGate: [UA.ARMORED, UA.STRUCTURE],
    sc2_units.Protoss.WarpPrism: [
        UA.ARMORED, UA.MECHANICAL, UA.PSIONIC, UA.FLYING
    ],
    sc2_units.Protoss.WarpPrismPhasing: [
        UA.ARMORED, UA.MECHANICAL, UA.PSIONIC, UA.FLYING
    ],
    sc2_units.Protoss.Zealot: [UA.LIGHT, UA.BIOLOGICAL],
    # Terran
    sc2_units.Terran.Armory: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.AutoTurret: [
        UA.MECHANICAL, UA.STRUCTURE, UA.ARMORED, UA.SUMMONED
    ],
    sc2_units.Terran.Banshee: [UA.LIGHT, UA.MECHANICAL, UA.FLYING],
    sc2_units.Terran.Barracks: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.BarracksFlying: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Terran.BarracksReactor: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.BarracksTechLab: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.Battlecruiser: [
        UA.ARMORED, UA.MECHANICAL, UA.MASSIVE, UA.FLYING
    ],
    sc2_units.Terran.Bunker: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.CommandCenter: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.CommandCenterFlying: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Terran.Cyclone: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Terran.EngineeringBay: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.Factory: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.FactoryFlying: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Terran.FactoryReactor: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.FactoryTechLab: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.FusionCore: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.Ghost: [UA.BIOLOGICAL, UA.PSIONIC],
    sc2_units.Terran.GhostAcademy: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.GhostAlternate: [UA.BIOLOGICAL, UA.PSIONIC],
    sc2_units.Terran.GhostNova: [UA.BIOLOGICAL, UA.PSIONIC],
    sc2_units.Terran.Hellion: [UA.LIGHT, UA.MECHANICAL],
    sc2_units.Terran.Hellbat: [UA.BIOLOGICAL, UA.LIGHT, UA.MECHANICAL],
    sc2_units.Terran.KD8Charge: [UA.SUMMONED],
    sc2_units.Terran.Liberator: [UA.ARMORED, UA.MECHANICAL, UA.FLYING],
    sc2_units.Terran.LiberatorAG: [UA.ARMORED, UA.MECHANICAL, UA.FLYING],
    sc2_units.Terran.MULE: [UA.LIGHT, UA.MECHANICAL, UA.SUMMONED],
    sc2_units.Terran.Marauder: [UA.ARMORED, UA.BIOLOGICAL],
    sc2_units.Terran.Marine: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Terran.Medivac: [UA.ARMORED, UA.MECHANICAL, UA.FLYING],
    sc2_units.Terran.MissileTurret: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.DETECTOR
    ],
    sc2_units.Terran.Nuke: [UA.SUMMONED],
    sc2_units.Terran.OrbitalCommand: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.OrbitalCommandFlying: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Terran.PlanetaryFortress: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL
    ],
    sc2_units.Terran.PointDefenseDrone: [
        UA.LIGHT, UA.MECHANICAL, UA.STRUCTURE, UA.SUMMONED
    ],
    sc2_units.Terran.Raven: [UA.LIGHT, UA.MECHANICAL, UA.DETECTOR, UA.FLYING],
    sc2_units.Terran.Reactor: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.Reaper: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Terran.Refinery: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.RepairDrone: [UA.SUMMONED],
    sc2_units.Terran.SCV: [UA.BIOLOGICAL, UA.LIGHT, UA.MECHANICAL],
    sc2_units.Terran.SensorTower: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.SiegeTank: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Terran.SiegeTankSieged: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Terran.Starport: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.StarportFlying: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.FLYING
    ],
    sc2_units.Terran.StarportReactor: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.StarportTechLab: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.SupplyDepot: [UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL],
    sc2_units.Terran.SupplyDepotLowered: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL
    ],
    sc2_units.Terran.TechLab: [
        UA.ARMORED, UA.STRUCTURE, UA.MECHANICAL, UA.ADDON
    ],
    sc2_units.Terran.Thor: [UA.ARMORED, UA.MECHANICAL, UA.MASSIVE],
    sc2_units.Terran.ThorHighImpactMode: [
        UA.ARMORED, UA.MECHANICAL, UA.MASSIVE
    ],
    sc2_units.Terran.VikingAssault: [UA.ARMORED, UA.MECHANICAL],
    sc2_units.Terran.VikingFighter: [UA.ARMORED, UA.MECHANICAL, UA.FLYING],
    sc2_units.Terran.WidowMine: [UA.MECHANICAL, UA.LIGHT],
    sc2_units.Terran.WidowMineBurrowed: [UA.MECHANICAL, UA.LIGHT, UA.BURROWED],
    # Zerg
    sc2_units.Zerg.Baneling: [UA.BIOLOGICAL],
    sc2_units.Zerg.BanelingBurrowed: [UA.BIOLOGICAL, UA.BURROWED],
    sc2_units.Zerg.BanelingCocoon: [UA.BIOLOGICAL],
    sc2_units.Zerg.BanelingNest: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.BroodLord: [
        UA.ARMORED, UA.BIOLOGICAL, UA.MASSIVE, UA.FLYING
    ],
    sc2_units.Zerg.BroodLordCocoon: [UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.Broodling: [UA.BIOLOGICAL, UA.LIGHT, UA.SUMMONED],
    sc2_units.Zerg.BroodlingEscort: [
        UA.BIOLOGICAL, UA.LIGHT, UA.FLYING, UA.SUMMONED
    ],
    sc2_units.Zerg.Changeling: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ChangelingMarine: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ChangelingMarineShield: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ChangelingZealot: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ChangelingZergling: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ChangelingZerglingWings: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.Cocoon: [UA.BIOLOGICAL],
    sc2_units.Zerg.Corruptor: [UA.ARMORED, UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.CreepTumor: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.CreepTumorBurrowed: [
        UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL, UA.BURROWED
    ],
    sc2_units.Zerg.CreepTumorQueen: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Drone: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.DroneBurrowed: [UA.BIOLOGICAL, UA.LIGHT, UA.BURROWED],
    sc2_units.Zerg.EvolutionChamber: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Extractor: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.GreaterSpire: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Hatchery: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Hive: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Hydralisk: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.HydraliskBurrowed: [UA.BIOLOGICAL, UA.LIGHT, UA.BURROWED],
    sc2_units.Zerg.HydraliskDen: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.InfestationPit: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.InfestedTerran: [UA.BIOLOGICAL, UA.LIGHT, UA.SUMMONED],
    sc2_units.Zerg.InfestedTerranBurrowed: [
        UA.BIOLOGICAL, UA.LIGHT, UA.SUMMONED, UA.BURROWED
    ],
    sc2_units.Zerg.InfestedTerranCocoon: [UA.BIOLOGICAL, UA.LIGHT, UA.SUMMONED],
    sc2_units.Zerg.Infestor: [UA.ARMORED, UA.BIOLOGICAL, UA.PSIONIC],
    sc2_units.Zerg.InfestorBurrowed: [
        UA.ARMORED, UA.BIOLOGICAL, UA.PSIONIC, UA.BURROWED
    ],
    sc2_units.Zerg.Lair: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Larva: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.Locust: [UA.LIGHT, UA.BIOLOGICAL],
    sc2_units.Zerg.LocustFlying: [UA.LIGHT, UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.Lurker: [UA.BIOLOGICAL, UA.ARMORED],
    sc2_units.Zerg.LurkerBurrowed: [UA.BIOLOGICAL, UA.ARMORED, UA.BURROWED],
    sc2_units.Zerg.LurkerDen: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.LurkerCocoon: [UA.BIOLOGICAL],
    sc2_units.Zerg.Mutalisk: [UA.BIOLOGICAL, UA.LIGHT, UA.FLYING],
    sc2_units.Zerg.NydusCanal: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.NydusNetwork: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Overlord: [UA.ARMORED, UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.OverlordTransport: [UA.ARMORED, UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.OverlordTransportCocoon: [UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.Overseer: [
        UA.ARMORED, UA.BIOLOGICAL, UA.FLYING, UA.DETECTOR
    ],
    sc2_units.Zerg.OverseerCocoon: [UA.BIOLOGICAL, UA.FLYING],
    sc2_units.Zerg.OverseerOversightMode: [
        UA.ARMORED, UA.BIOLOGICAL, UA.FLYING, UA.DETECTOR
    ],
    sc2_units.Zerg.ParasiticBombDummy: [UA.SUMMONED],
    sc2_units.Zerg.Queen: [UA.BIOLOGICAL, UA.PSIONIC],
    sc2_units.Zerg.QueenBurrowed: [UA.BIOLOGICAL, UA.PSIONIC, UA.BURROWED],
    sc2_units.Zerg.Ravager: [UA.BIOLOGICAL],
    sc2_units.Zerg.RavagerBurrowed: [UA.BIOLOGICAL, UA.BURROWED],
    sc2_units.Zerg.RavagerCocoon: [UA.BIOLOGICAL],
    sc2_units.Zerg.Roach: [UA.ARMORED, UA.BIOLOGICAL],
    sc2_units.Zerg.RoachBurrowed: [UA.ARMORED, UA.BIOLOGICAL, UA.BURROWED],
    sc2_units.Zerg.RoachWarren: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.SpawningPool: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.SpineCrawler: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.SpineCrawlerUprooted: [
        UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL
    ],
    sc2_units.Zerg.Spire: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.SporeCrawler: [
        UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL, UA.DETECTOR
    ],
    sc2_units.Zerg.SporeCrawlerUprooted: [
        UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL
    ],
    sc2_units.Zerg.SwarmHost: [UA.ARMORED, UA.BIOLOGICAL],
    sc2_units.Zerg.SwarmHostBurrowed: [UA.ARMORED, UA.BIOLOGICAL, UA.BURROWED],
    sc2_units.Zerg.Ultralisk: [UA.ARMORED, UA.BIOLOGICAL, UA.MASSIVE],
    sc2_units.Zerg.UltraliskBurrowed: [
        UA.ARMORED, UA.BIOLOGICAL, UA.MASSIVE, UA.BURROWED
    ],
    sc2_units.Zerg.UltraliskCavern: [UA.ARMORED, UA.STRUCTURE, UA.BIOLOGICAL],
    sc2_units.Zerg.Viper: [UA.ARMORED, UA.BIOLOGICAL, UA.PSIONIC, UA.FLYING],
    sc2_units.Zerg.Zergling: [UA.BIOLOGICAL, UA.LIGHT],
    sc2_units.Zerg.ZerglingBurrowed: [UA.BIOLOGICAL, UA.LIGHT, UA.BURROWED],
}


def get_attribute_lookup(num_unit_types: int) -> chex.Array:
  """Returns an boolean array specifying the attributes of each unit."""
  attribute_lookup = np.zeros((num_unit_types, max(UA).value + 1),
                              dtype=np.float32)
  for unit, attributes in UNITS_ATTRIBUTES.items():
    unit_id = uint8_lookup.PySc2ToUint8(unit)
    for attribute in attributes:
      attribute_lookup[unit_id, attribute.value] = 1.
  return attribute_lookup


def get_order_id_lookup(function_names: List[str],
                        redundant_list: Optional[Mapping[str, str]] = None
                        ) -> np.ndarray:
  """Remaps function arguments to remove redundent ones."""
  if redundant_list is None:
    redundant_list = dict(REDUNDANT_GENERIC_ORDER_ID)
  current_max = 0
  lookup_map = {}
  for i, function in enumerate(function_names):
    if function not in redundant_list:
      lookup_map[i] = current_max
      current_max += 1
  lookup = np.zeros((len(function_names),), dtype=np.int32)
  for i, function in enumerate(function_names):
    if function in redundant_list:
      remapped_name = redundant_list[function]
      lookup[i] = lookup_map[function_names.index(remapped_name)]
    else:
      lookup[i] = lookup_map[i]
  return lookup


def get_build_queue_order_id_lookup(function_names: List[str]) -> np.ndarray:
  """Remaps function arguments to remove irrelevant ones to build queue."""
  remap = dict(REDUNDANT_GENERIC_ORDER_ID)  # copy
  for fun in sc2_actions.RAW_FUNCTIONS:
    name = fun.name
    if (name == "no_op") or ("Train_" in name) or ("Research_" in name):
      continue
    else:
      remap[name] = "no_op"
  return get_order_id_lookup(function_names, remap)


def get_addon_lookup(num_unit_types: int) -> np.ndarray:
  """Remaps units to keep only the add-on types."""
  lookup = np.zeros((num_unit_types,), dtype=np.int32)
  for i, unit in enumerate(ADDON_UNIT_TYPES):
    lookup[uint8_lookup.PySc2ToUint8(unit)] = i + 1
  return lookup
