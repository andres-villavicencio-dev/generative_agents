"""
Tests for concurrent resource contention handling.

Tests the scenario where multiple agents try to use shared resources
(like showers/hot water) simultaneously.
"""
import sys
import os
import unittest
import tempfile
import shutil
import json

# Add backend server to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server'))

from resource_manager import (
    WorldResourceManager,
    get_persona_decay_multiplier,
    PERSONA_NEED_BASELINES,
    DEFAULT_PERSONA_BASELINES,
)


class TestResourceContention(unittest.TestCase):
    """Test concurrent access to shared resources like showers."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary simulation folder
        self.temp_dir = tempfile.mkdtemp()
        self.sim_folder = os.path.join(self.temp_dir, "test_sim")
        os.makedirs(self.sim_folder)

        # Initialize resource manager
        self.rm = WorldResourceManager(self.sim_folder)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_single_agent_acquires_shower(self):
        """Test that a single agent can acquire a shower."""
        success, wait, locked_by = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )
        self.assertTrue(success)
        self.assertEqual(wait, 0)
        self.assertIsNone(locked_by)

    def test_second_agent_blocked_from_shower(self):
        """Test that a second agent is blocked when shower is in use."""
        # First agent acquires
        success1, _, _ = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )
        self.assertTrue(success1)

        # Second agent tries to acquire - should fail
        success2, wait, locked_by = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Isabella Rodriguez",
            current_tick=1
        )
        self.assertFalse(success2)
        self.assertGreater(wait, 0)
        self.assertEqual(locked_by, "Klaus Mueller")

    def test_five_agents_one_shower(self):
        """Test that only one of five agents can use the shower at a time."""
        agents = [
            "Klaus Mueller",
            "Isabella Rodriguez",
            "Maria Lopez",
            "John Lin",
            "Mei Lin"
        ]

        successes = []
        failures = []

        # All five agents try to acquire at tick 0
        for agent in agents:
            success, wait, locked_by = self.rm.try_acquire(
                "common bathroom:bathroom:hot_water",
                "hot_water",
                agent,
                current_tick=0
            )
            if success:
                successes.append(agent)
            else:
                failures.append((agent, locked_by))

        # Exactly one agent should succeed
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 4)

        # All failures should report the successful agent as the blocker
        winner = successes[0]
        for agent, locked_by in failures:
            self.assertEqual(locked_by, winner)

    def test_lock_expires_after_duration(self):
        """Test that locks expire after their duration."""
        # First agent acquires at tick 0
        self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )

        # Default duration is 3 ticks, so at tick 3 lock should be expired
        # Second agent should be able to acquire
        success, wait, locked_by = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Isabella Rodriguez",
            current_tick=3
        )
        self.assertTrue(success)
        self.assertEqual(wait, 0)

    def test_explicit_release(self):
        """Test that explicit release frees the resource."""
        # Acquire
        self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )

        # Release early
        self.rm.release(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller"
        )

        # Another agent should be able to acquire immediately
        success, wait, locked_by = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Isabella Rodriguez",
            current_tick=1
        )
        self.assertTrue(success)

    def test_wrong_persona_cannot_release(self):
        """Test that only the lock owner can release."""
        # Klaus acquires
        self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )

        # Isabella tries to release Klaus's lock
        self.rm.release(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Isabella Rodriguez"
        )

        # Lock should still be held by Klaus
        success, wait, locked_by = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Maria Lopez",
            current_tick=1
        )
        self.assertFalse(success)
        self.assertEqual(locked_by, "Klaus Mueller")

    def test_non_shared_resource_always_succeeds(self):
        """Test that non-shared resources don't block."""
        # coffee_beans is not a shared resource
        success, wait, locked_by = self.rm.try_acquire(
            "Hobbs Cafe:kitchen:coffee machine",
            "coffee",  # Not in SHARED_RESOURCES
            "Klaus Mueller",
            current_tick=0
        )
        self.assertTrue(success)
        self.assertEqual(wait, 0)
        self.assertIsNone(locked_by)

    def test_different_locations_independent(self):
        """Test that different bathroom locations are independent."""
        # Klaus uses common bathroom
        success1, _, _ = self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )
        self.assertTrue(success1)

        # Isabella uses her own bathroom - should succeed
        success2, wait, locked_by = self.rm.try_acquire(
            "Isabella Rodriguez's apartment:bathroom:hot_water",
            "hot_water",
            "Isabella Rodriguez",
            current_tick=0
        )
        self.assertTrue(success2)
        self.assertEqual(wait, 0)

    def test_get_lock_status(self):
        """Test getting lock status."""
        # Initially no lock
        status = self.rm.get_lock_status(
            "common bathroom:bathroom:hot_water",
            "hot_water"
        )
        self.assertIsNone(status)

        # After acquiring
        self.rm.try_acquire(
            "common bathroom:bathroom:hot_water",
            "hot_water",
            "Klaus Mueller",
            current_tick=0
        )

        status = self.rm.get_lock_status(
            "common bathroom:bathroom:hot_water",
            "hot_water"
        )
        self.assertIsNotNone(status)
        self.assertEqual(status["locked_by"], "Klaus Mueller")


class TestPersonaBaselines(unittest.TestCase):
    """Test persona-specific need decay baselines."""

    def test_klaus_mueller_baselines(self):
        """Test Klaus Mueller's specific baselines."""
        # Klaus is a coffee addict with high thirst decay
        thirst_mult = get_persona_decay_multiplier("Klaus Mueller", "thirst")
        self.assertEqual(thirst_mult, 1.5)

        # Klaus is introverted, lower social decay
        social_mult = get_persona_decay_multiplier("Klaus Mueller", "social")
        self.assertEqual(social_mult, 0.8)

        # Klaus needs mental stimulation
        stim_mult = get_persona_decay_multiplier("Klaus Mueller", "stimulation")
        self.assertEqual(stim_mult, 1.3)

    def test_isabella_rodriguez_baselines(self):
        """Test Isabella Rodriguez's specific baselines."""
        # Isabella stays hydrated (cafe owner)
        thirst_mult = get_persona_decay_multiplier("Isabella Rodriguez", "thirst")
        self.assertEqual(thirst_mult, 0.7)

        # Isabella is a social butterfly
        social_mult = get_persona_decay_multiplier("Isabella Rodriguez", "social")
        self.assertEqual(social_mult, 1.3)

    def test_maria_lopez_baselines(self):
        """Test Maria Lopez's specific baselines."""
        # Maria has higher hygiene standard
        hygiene_mult = get_persona_decay_multiplier("Maria Lopez", "hygiene")
        self.assertEqual(hygiene_mult, 1.2)

    def test_unknown_persona_gets_defaults(self):
        """Test that unknown personas get default multipliers."""
        mult = get_persona_decay_multiplier("Unknown Person", "hunger")
        self.assertEqual(mult, 1.0)

        mult = get_persona_decay_multiplier("Random Agent", "thirst")
        self.assertEqual(mult, 1.0)

    def test_unknown_need_type_returns_one(self):
        """Test that unknown need types return 1.0."""
        mult = get_persona_decay_multiplier("Klaus Mueller", "unknown_need")
        self.assertEqual(mult, 1.0)


class TestWorldResourceManager(unittest.TestCase):
    """Test WorldResourceManager basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sim_folder = os.path.join(self.temp_dir, "test_sim")
        os.makedirs(self.sim_folder)
        self.rm = WorldResourceManager(self.sim_folder)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Test that resource state persists across saves."""
        # Modify some state
        self.rm.consume("common bathroom:bathroom:hot_water", "level", 20)
        original_level = self.rm.get_stock("common bathroom:bathroom:hot_water", "level")

        # Save
        self.rm.save()

        # Create new manager from same folder
        rm2 = WorldResourceManager(self.sim_folder)
        loaded_level = rm2.get_stock("common bathroom:bathroom:hot_water", "level")

        self.assertEqual(original_level, loaded_level)

    def test_consume_resource(self):
        """Test resource consumption."""
        initial = self.rm.get_stock("common bathroom:bathroom:hot_water", "level")
        self.rm.consume("common bathroom:bathroom:hot_water", "level", 15)
        after = self.rm.get_stock("common bathroom:bathroom:hot_water", "level")

        self.assertEqual(after, initial - 15)

    def test_consume_fails_on_insufficient(self):
        """Test that consumption fails when insufficient."""
        # Try to consume more than available
        success = self.rm.consume("common bathroom:bathroom:hot_water", "level", 1000)
        self.assertFalse(success)

    def test_restock_resource(self):
        """Test restocking resources."""
        # Consume first
        self.rm.consume("common bathroom:bathroom:hot_water", "level", 50)
        mid = self.rm.get_stock("common bathroom:bathroom:hot_water", "level")

        # Restock
        self.rm.restock("common bathroom:bathroom:hot_water", "level", 30)
        after = self.rm.get_stock("common bathroom:bathroom:hot_water", "level")

        self.assertEqual(after, mid + 30)


if __name__ == '__main__':
    unittest.main()
