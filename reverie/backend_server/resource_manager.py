"""
File: resource_manager.py
Description: Manages world resources for the Generative Agents simulation.
This implements Phase 2 of the needs-and-resources extension.

Resources include consumables (food, coffee beans, hot water) at various
locations in the world. The ResourceManager tracks stock levels, handles
consumption, passive refills, and daily deliveries.
"""
import json
import os
import datetime
import threading


# Item prices for economy layer (Phase 5)
ITEM_PRICES = {
    "eggs": 0.50,       # per unit
    "bread": 1.50,      # per loaf (unit)
    "milk": 1.20,       # per unit
    "coffee_beans": 2.00,
    "sandwiches": 4.50,
    "pastries": 3.00,
}


class WorldResourceManager:
    """
    Manages world resources across all locations in the simulation.

    Resources are keyed by address strings like:
    "Isabella Rodriguez's apartment:kitchen:refrigerator"

    Each resource location contains items with quantities, and optionally
    max capacities and refill rates for continuous resources like hot water.
    """

    # Default world state for first-run initialization
    DEFAULT_WORLD_STATE = {
        "Isabella Rodriguez's apartment:kitchen:refrigerator": {
            "eggs": 6, "bread": 2, "milk": 1.0, "coffee_beans": 0.5,
            "last_restocked": "February 13, 2023, 06:00:00"
        },
        "Klaus Mueller's room:kitchen:refrigerator": {
            "eggs": 2, "bread": 1, "milk": 0.5, "coffee_beans": 0.0,
            "last_restocked": "February 12, 2023, 18:00:00"
        },
        "Maria Lopez's apartment:kitchen:refrigerator": {
            "eggs": 4, "bread": 0, "milk": 0.3, "coffee_beans": 1.0,
            "last_restocked": "February 12, 2023, 20:00:00"
        },
        "Hobbs Cafe:kitchen:coffee machine": {
            "coffee_beans": 80.0, "max": 100
        },
        "Hobbs Cafe:front counter:prepared_food": {
            "sandwiches": 0, "pastries": 0, "max": 20
        },
        "the Ville:Hobbs Cafe:cafe:coffee machine": {
            "coffee_beans": 80.0, "max": 100
        },
        "the Ville:Hobbs Cafe:cafe:counter": {
            "sandwiches": 0, "pastries": 0, "max": 20
        },
        "store:shelves:groceries": {
            "eggs": 100, "bread": 50, "milk": 40, "coffee_beans": 30,
            "last_delivery": "February 13, 2023, 06:00:00"
        },
        "the Ville:Harvey Oak Supply Store:supply store:shelves": {
            "eggs": 100, "bread": 50, "milk": 40, "coffee_beans": 30,
            "last_delivery": "February 13, 2023, 06:00:00"
        },
        "common bathroom:bathroom:hot_water": {
            "level": 80.0, "max": 100, "refill_rate": 1.0
        },
        "Isabella Rodriguez's apartment:bathroom:hot_water": {
            "level": 100.0, "max": 100, "refill_rate": 1.0
        },
        "Klaus Mueller's room:bathroom:hot_water": {
            "level": 100.0, "max": 100, "refill_rate": 1.0
        },
        "Maria Lopez's apartment:bathroom:hot_water": {
            "level": 100.0, "max": 100, "refill_rate": 1.0
        },
        # Additional bathroom locations that may exist in the ville
        "the Ville:Dorm for Oak Hill College:bathroom:hot_water": {
            "level": 100.0, "max": 100, "refill_rate": 1.0
        }
    }

    # Shared resources that can only be used by one agent at a time
    SHARED_RESOURCES = {
        "hot_water": {"capacity": 1, "use_duration_ticks": 3},
        "shower": {"capacity": 1, "use_duration_ticks": 3},
        "bathroom": {"capacity": 1, "use_duration_ticks": 2},
    }

    # Store locations for daily restocking
    STORE_ADDRESSES = [
        "store:shelves:groceries",
        "the Ville:Harvey Oak Supply Store:supply store:shelves"
    ]

    # Store max stock levels
    STORE_MAX_STOCK = {
        "eggs": 100,
        "bread": 50,
        "milk": 40,
        "coffee_beans": 30
    }

    def __init__(self, sim_folder):
        """
        Initialize the ResourceManager.

        Args:
            sim_folder: Path to the simulation folder (e.g., storage/sim_code)
        """
        self.sim_folder = sim_folder
        self.resources_file = os.path.join(sim_folder, "resources", "world_state.json")
        self.last_tick_time = None
        self.last_delivery_date = None

        # Load existing state or initialize with defaults
        if os.path.exists(self.resources_file):
            try:
                with open(self.resources_file, "r") as f:
                    self.world_state = json.load(f)
            except (json.JSONDecodeError, IOError):
                print("[ResourceManager] Warning: Could not load world_state.json, using defaults")
                self.world_state = dict(self.DEFAULT_WORLD_STATE)
        else:
            print("[ResourceManager] Initializing with default world state")
            self.world_state = dict(self.DEFAULT_WORLD_STATE)

        # Resource locks for shared resources (concurrent contention)
        # Format: {address: {"locked_by": persona_name, "until_tick": tick_number}}
        self.resource_locks = {}
        self._lock_mutex = threading.Lock()
        self.current_tick = 0

    def _normalize_address(self, address):
        """
        Normalize an address for matching. Handles partial matches.
        Returns the best matching key from world_state, or None.
        """
        address_lower = address.lower()

        # Direct match first
        if address in self.world_state:
            return address

        # Try case-insensitive match
        for key in self.world_state:
            if key.lower() == address_lower:
                return key

        # Try partial/fuzzy matching - check if address is contained in key or vice versa
        for key in self.world_state:
            key_lower = key.lower()
            # Check if address components match
            if address_lower in key_lower or key_lower in address_lower:
                return key
            # Check if last component (object) matches
            addr_parts = address_lower.split(":")
            key_parts = key_lower.split(":")
            if addr_parts and key_parts and addr_parts[-1] == key_parts[-1]:
                return key

        return None

    def get_stock(self, address, item):
        """
        Get the current stock level of an item at an address.

        Args:
            address: Location address string
            item: Item name (e.g., "eggs", "coffee_beans", "level" for hot water)

        Returns:
            float: Stock level, or 0.0 if not found
        """
        normalized = self._normalize_address(address)
        if normalized and normalized in self.world_state:
            return self.world_state[normalized].get(item, 0.0)
        return 0.0

    def consume(self, address, item, amount):
        """
        Consume a resource at the given address.

        Args:
            address: Location address string
            item: Item name to consume
            amount: Amount to consume

        Returns:
            bool: True if consumption succeeded, False if insufficient stock
        """
        normalized = self._normalize_address(address)
        if not normalized or normalized not in self.world_state:
            # Address not tracked - assume infinite supply (backwards compatibility)
            return True

        location = self.world_state[normalized]
        current = location.get(item, 0.0)

        if current < amount:
            return False

        location[item] = current - amount
        return True

    def purchase(self, address, item, amount, buyer_scratch):
        """
        Consume stock AND deduct from buyer wallet.

        Args:
            address: Location address string
            item: Item name to purchase
            amount: Amount to purchase
            buyer_scratch: Buyer's Scratch object (must have .wallet attribute)

        Returns:
            bool: True if purchase succeeded (stock + funds available), False otherwise
        """
        price = ITEM_PRICES.get(item, 1.0) * amount

        # Check if buyer has enough funds
        if not hasattr(buyer_scratch, 'wallet') or buyer_scratch.wallet < price:
            return False

        # Check if stock is available and consume it
        if not self.consume(address, item, amount):
            return False

        # Deduct from wallet
        buyer_scratch.wallet -= price

        # Update financial stress based on remaining funds
        if hasattr(buyer_scratch, 'financial_stress'):
            if buyer_scratch.wallet < 20:
                buyer_scratch.financial_stress = min(1.0, buyer_scratch.financial_stress + 0.1)
            elif buyer_scratch.wallet < 50:
                buyer_scratch.financial_stress = min(1.0, buyer_scratch.financial_stress + 0.05)

        return True

    def restock(self, address, item, amount):
        """
        Add stock to a resource at the given address.

        Args:
            address: Location address string
            item: Item name to restock
            amount: Amount to add
        """
        normalized = self._normalize_address(address)
        if not normalized:
            # Create new entry if address doesn't exist
            normalized = address
            self.world_state[normalized] = {}

        location = self.world_state[normalized]
        current = location.get(item, 0.0)
        max_val = location.get("max", float('inf'))

        location[item] = min(current + amount, max_val)

    def tick(self, sim_time):
        """
        Process passive resource changes each simulation tick.

        - Refills hot_water by refill_rate each tick (capped at max)
        - At 6:00 AM each sim day, restocks store:shelves:groceries to full

        Args:
            sim_time: Current simulation datetime
        """
        try:
            # Hot water refill - happens every tick
            for address, location in self.world_state.items():
                if "hot_water" in address.lower() or "hot_water" in str(location):
                    if "level" in location and "refill_rate" in location:
                        max_level = location.get("max", 100.0)
                        refill_rate = location.get("refill_rate", 1.0)
                        location["level"] = min(location["level"] + refill_rate, max_level)

            # Daily store delivery at 6:00 AM
            current_date = sim_time.date()
            current_hour = sim_time.hour

            # Check if it's 6:00 AM and we haven't delivered today
            if current_hour == 6:
                if self.last_delivery_date != current_date:
                    self._do_daily_delivery(sim_time)
                    self.last_delivery_date = current_date

            self.last_tick_time = sim_time

        except Exception as e:
            print(f"[ResourceManager] Error in tick: {e}")

    def _do_daily_delivery(self, sim_time):
        """
        Perform daily store restocking.
        """
        time_str = sim_time.strftime("%B %d, %Y, %H:%M:%S")

        for store_addr in self.STORE_ADDRESSES:
            normalized = self._normalize_address(store_addr)
            if normalized and normalized in self.world_state:
                location = self.world_state[normalized]
                for item, max_stock in self.STORE_MAX_STOCK.items():
                    location[item] = max_stock
                location["last_delivery"] = time_str
                print(f"[ResourceManager] Daily delivery to {normalized}")

    def try_acquire(self, address, resource_type, persona_name, current_tick):
        """
        Try to acquire exclusive access to a shared resource.

        Args:
            address: Location address string (e.g., "common bathroom:bathroom:hot_water")
            resource_type: Type of shared resource (e.g., "hot_water", "shower")
            persona_name: Name of the persona trying to acquire
            current_tick: Current simulation tick number

        Returns:
            tuple: (success: bool, wait_ticks: int, locked_by: str or None)
        """
        # Check if this is a shared resource type
        if resource_type not in self.SHARED_RESOURCES:
            return (True, 0, None)  # Not a shared resource, always succeeds

        with self._lock_mutex:
            self.current_tick = current_tick

            # Clean up expired locks
            self._cleanup_expired_locks(current_tick)

            # Check if resource is currently locked
            lock_key = f"{address}:{resource_type}"
            if lock_key in self.resource_locks:
                lock_info = self.resource_locks[lock_key]
                if lock_info["until_tick"] > current_tick:
                    # Resource is locked by someone else
                    wait_ticks = lock_info["until_tick"] - current_tick
                    return (False, wait_ticks, lock_info["locked_by"])

            # Acquire the lock
            duration = self.SHARED_RESOURCES[resource_type]["use_duration_ticks"]
            self.resource_locks[lock_key] = {
                "locked_by": persona_name,
                "until_tick": current_tick + duration
            }
            return (True, 0, None)

    def release(self, address, resource_type, persona_name):
        """
        Release a resource lock early (e.g., if action is interrupted).

        Args:
            address: Location address string
            resource_type: Type of shared resource
            persona_name: Name of the persona releasing
        """
        with self._lock_mutex:
            lock_key = f"{address}:{resource_type}"
            if lock_key in self.resource_locks:
                lock_info = self.resource_locks[lock_key]
                # Only release if this persona owns the lock
                if lock_info["locked_by"] == persona_name:
                    del self.resource_locks[lock_key]

    def _cleanup_expired_locks(self, current_tick):
        """Remove expired locks."""
        expired = [k for k, v in self.resource_locks.items()
                   if v["until_tick"] <= current_tick]
        for k in expired:
            del self.resource_locks[k]

    def get_lock_status(self, address, resource_type):
        """
        Check if a resource is currently locked.

        Returns:
            dict or None: Lock info if locked, None if available
        """
        lock_key = f"{address}:{resource_type}"
        with self._lock_mutex:
            self._cleanup_expired_locks(self.current_tick)
            return self.resource_locks.get(lock_key)

    def save(self, sim_folder=None):
        """
        Save the current world state to disk.

        Args:
            sim_folder: Optional override for sim folder path
        """
        try:
            folder = sim_folder or self.sim_folder
            resources_dir = os.path.join(folder, "resources")
            os.makedirs(resources_dir, exist_ok=True)

            out_file = os.path.join(resources_dir, "world_state.json")
            with open(out_file, "w") as f:
                json.dump(self.world_state, f, indent=2)
        except Exception as e:
            print(f"[ResourceManager] Error saving world state: {e}")

    def get_str_resource_context(self, persona_name, location):
        """
        Generate a natural language description of relevant resources at a location.
        Used for LLM context injection during planning.

        Args:
            persona_name: Name of the persona (e.g., "Isabella Rodriguez")
            location: Current location address string

        Returns:
            str: Natural language resource description, or "" if no relevant resources
        """
        if not location:
            return ""

        location_lower = location.lower()
        context_parts = []

        # Get persona's first name for home matching
        persona_first = persona_name.split()[0].lower() if persona_name else ""

        # Find relevant resources based on location
        for address, items in self.world_state.items():
            address_lower = address.lower()

            # Determine if this is the persona's home resource
            is_home_resource = persona_first and persona_first in address_lower

            # Check if this resource location matches current location
            location_match = False

            # For home resources, only match if persona is at home
            if is_home_resource:
                if persona_first in location_lower:
                    location_match = True
                # Also match if just looking at "kitchen" and this is persona's kitchen
                elif "kitchen" in location_lower and "kitchen" in address_lower:
                    location_match = True
            else:
                # For non-home resources, check direct location match
                if location_lower in address_lower or address_lower in location_lower:
                    location_match = True
                # Check if at cafe
                elif "cafe" in location_lower and "cafe" in address_lower:
                    location_match = True
                elif "hobbs" in location_lower and "hobbs" in address_lower:
                    location_match = True
                # Check if at store
                elif "store" in location_lower and "store" in address_lower:
                    location_match = True

            if not location_match:
                continue

            # Generate description based on resource type
            if "refrigerator" in address_lower or "fridge" in address_lower:
                # Only report fridge contents for persona's own home
                if not is_home_resource:
                    continue
                fridge_items = []
                for item, qty in items.items():
                    if item in ("last_restocked", "max"):
                        continue
                    if isinstance(qty, (int, float)):
                        if qty <= 0:
                            fridge_items.append(f"no {item}")
                        elif qty < 1:
                            fridge_items.append(f"{item} (low)")
                        else:
                            fridge_items.append(f"{item} ×{int(qty)}")

                if fridge_items:
                    context_parts.append(f"Your fridge has: {', '.join(fridge_items)}.")

            elif "coffee machine" in address_lower or "coffee_machine" in address_lower:
                beans = items.get("coffee_beans", 0)
                max_beans = items.get("max", 100)
                pct = int((beans / max_beans) * 100) if max_beans > 0 else 0
                if "cafe" in address_lower or "hobbs" in address_lower:
                    context_parts.append(f"The café has {pct}% coffee beans stock.")
                else:
                    context_parts.append(f"Coffee machine has {pct}% beans.")

            elif "hot_water" in address_lower:
                level = items.get("level", 100)
                max_level = items.get("max", 100)
                pct = int((level / max_level) * 100) if max_level > 0 else 0
                if pct < 50:
                    context_parts.append(f"Hot water is low ({pct}%).")

            elif "prepared_food" in address_lower or "counter" in address_lower:
                sandwiches = items.get("sandwiches", 0)
                pastries = items.get("pastries", 0)
                if "cafe" in address_lower or "hobbs" in address_lower:
                    if sandwiches > 0 or pastries > 0:
                        context_parts.append(f"Café counter has {int(sandwiches)} sandwiches, {int(pastries)} pastries.")
                    else:
                        context_parts.append("Café counter needs restocking.")

            elif "shelves" in address_lower or "groceries" in address_lower:
                if "store" in address_lower or "supply" in address_lower:
                    stock_items = []
                    for item, qty in items.items():
                        if item in ("last_delivery", "max"):
                            continue
                        if isinstance(qty, (int, float)) and qty > 0:
                            stock_items.append(f"{item} ({int(qty)})")
                    if stock_items:
                        context_parts.append(f"Store has: {', '.join(stock_items)}.")

        # Deduplicate context parts
        seen = set()
        unique_parts = []
        for part in context_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        return " ".join(unique_parts)

    def find_nearest_resource(self, persona_location, resource_type):
        """
        Find the nearest location with a specific resource type.

        Args:
            persona_location: Current persona location address
            resource_type: Type of resource needed (e.g., "refrigerator", "coffee_beans")

        Returns:
            str: Address of nearest matching resource, or None
        """
        # Simple implementation - just returns first match
        # Could be extended with actual distance calculation
        for address in self.world_state:
            if resource_type.lower() in address.lower():
                return address
            items = self.world_state[address]
            if resource_type in items and items[resource_type] > 0:
                return address
        return None

    def _update_financial_stress(self, scratch):
        """Update financial stress based on wallet level (Phase 5)."""
        if not hasattr(scratch, 'wallet'):
            return
        w = scratch.wallet
        if w < 10:
            scratch.financial_stress = 1.0  # max stress
        elif w < 30:
            scratch.financial_stress = 0.7
        elif w < 60:
            scratch.financial_stress = 0.4
        else:
            scratch.financial_stress = 0.0


# Action to resource consumption mappings
# Format: { keyword_in_action: [(resource_address_pattern, item, amount), ...] }
ACTION_RESOURCE_MAPPINGS = {
    # Eating actions
    "eating": [("refrigerator", "eggs", 1), ("refrigerator", "bread", 0.5)],
    "breakfast": [("refrigerator", "eggs", 2), ("refrigerator", "bread", 1), ("refrigerator", "milk", 0.2)],
    "lunch": [("refrigerator", "bread", 1), ("counter", "sandwiches", 1)],
    "dinner": [("refrigerator", "eggs", 2), ("refrigerator", "bread", 1)],
    "meal": [("refrigerator", "eggs", 1), ("refrigerator", "bread", 1)],

    # Coffee actions
    "coffee": [("coffee", "coffee_beans", 0.1), ("refrigerator", "coffee_beans", 0.1)],
    "making coffee": [("coffee", "coffee_beans", 0.2)],

    # Hygiene actions
    "shower": [("hot_water", "level", 15)],
    "showering": [("hot_water", "level", 15)],
    "bath": [("hot_water", "level", 25)],

    # Cafe preparation (Isabella)
    "preparing cafe": [("refrigerator", "eggs", 4), ("refrigerator", "bread", 2)],
    "preparing food": [("refrigerator", "eggs", 2), ("refrigerator", "bread", 1)],
}

# Production mappings - when consuming resources produces other resources
PRODUCTION_MAPPINGS = {
    "preparing cafe": [("counter", "sandwiches", 5), ("counter", "pastries", 5)],
    "preparing food": [("counter", "sandwiches", 2), ("counter", "pastries", 2)],
    "preparing pastries": [("counter", "sandwiches", 2), ("counter", "pastries", 3)],
    "baked goods": [("counter", "sandwiches", 2), ("counter", "pastries", 3)],
    "opening the cafe": [("counter", "sandwiches", 3), ("counter", "pastries", 3)],
    "open hobbs": [("counter", "sandwiches", 3), ("counter", "pastries", 3)],
    "preparing the cafe": [("counter", "sandwiches", 3), ("counter", "pastries", 3)],
    "open cafe": [("counter", "sandwiches", 3), ("counter", "pastries", 3)],
}

# Persona-specific need decay rates and starting values
# Different characters have different metabolic/personality traits
PERSONA_NEED_BASELINES = {
    "Klaus Mueller": {
        "hunger_decay_mult": 0.75,       # Less hungry (disciplined researcher)
        "thirst_decay_mult": 1.5,        # Coffee addict — needs drinks often
        "energy_decay_mult": 1.0,        # Normal energy
        "hygiene_decay_mult": 1.0,       # Normal hygiene
        "bladder_decay_mult": 1.2,       # Drinks more -> more bathroom
        "social_decay_mult": 0.8,        # Introverted, less social need
        "comfort_decay_mult": 1.0,       # Normal
        "stimulation_decay_mult": 1.3,   # Needs mental stimulation
    },
    "Isabella Rodriguez": {
        "hunger_decay_mult": 1.0,        # Normal hunger
        "thirst_decay_mult": 0.7,        # Stays hydrated (café owner)
        "energy_decay_mult": 0.8,        # High stamina from work
        "hygiene_decay_mult": 0.8,       # Professional appearance
        "bladder_decay_mult": 1.0,       # Normal
        "social_decay_mult": 1.3,        # Social butterfly, needs interaction
        "comfort_decay_mult": 1.0,       # Normal
        "stimulation_decay_mult": 0.9,   # Gets stimulation from customers
    },
    "Maria Lopez": {
        "hunger_decay_mult": 1.0,        # Normal
        "thirst_decay_mult": 1.0,        # Normal
        "energy_decay_mult": 0.9,        # Writer stamina
        "hygiene_decay_mult": 1.2,       # Higher hygiene standard
        "bladder_decay_mult": 1.0,       # Normal
        "social_decay_mult": 1.1,        # Moderate social need
        "comfort_decay_mult": 0.8,       # Comfortable working from home
        "stimulation_decay_mult": 1.1,   # Needs creative stimulation
    },
}

# Default multipliers for personas not in the list above
DEFAULT_PERSONA_BASELINES = {
    "hunger_decay_mult": 1.0,
    "thirst_decay_mult": 1.0,
    "energy_decay_mult": 1.0,
    "hygiene_decay_mult": 1.0,
    "bladder_decay_mult": 1.0,
    "social_decay_mult": 1.0,
    "comfort_decay_mult": 1.0,
    "stimulation_decay_mult": 1.0,
}


def get_persona_decay_multiplier(persona_name, need_type):
    """
    Get the decay rate multiplier for a specific persona and need type.

    Args:
        persona_name: Full name of the persona (e.g., "Klaus Mueller")
        need_type: The need type (e.g., "hunger", "thirst", etc.)

    Returns:
        float: Multiplier to apply to the base decay rate
    """
    baselines = PERSONA_NEED_BASELINES.get(persona_name, DEFAULT_PERSONA_BASELINES)
    key = f"{need_type}_decay_mult"
    return baselines.get(key, 1.0)
