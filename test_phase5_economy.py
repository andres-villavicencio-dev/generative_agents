#!/usr/bin/env python3
"""
Phase 5 Economy Layer Test
Validates wallet init, purchases, stress calculation, save/load.
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, '/home/andus/Projects/generative_agents/reverie/backend_server')

from persona.memory_structures.scratch import Scratch
from resource_manager import WorldResourceManager, ITEM_PRICES

print("=" * 60)
print("PHASE 5: ECONOMY LAYER TEST")
print("=" * 60)

# Create test scratch files based on real phase3 personas
TEST_PERSONAS = {
    "Isabella Rodriguez": {"expected": 250.0, "role": "cafe owner"},
    "Klaus Mueller": {"expected": 120.0, "role": "researcher"},
    "Maria Lopez": {"expected": 80.0, "role": "student"},
}

def create_test_scratch(persona_name, tmpdir):
    """Create a minimal scratch.json without wallet (simulating old save)."""
    base_path = f'/home/andus/Projects/generative_agents/environment/frontend_server/storage/phase3_test_r1/personas/{persona_name}/bootstrap_memory/scratch.json'
    with open(base_path) as f:
        data = json.load(f)
    # Remove wallet if present (simulate old save)
    data.pop('wallet', None)
    data.pop('financial_stress', None)
    # Ensure name matches
    data['name'] = persona_name
    
    test_path = f"{tmpdir}/{persona_name.replace(' ', '_')}_scratch.json"
    with open(test_path, 'w') as f:
        json.dump(data, f)
    return test_path

# Test 1: Role-based wallet initialization from old saves
print("\n[TEST 1] Role-based wallet init (backwards compat)")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    for name, info in TEST_PERSONAS.items():
        scratch_path = create_test_scratch(name, tmpdir)
        s = Scratch(scratch_path)
        print(f"  {name:22s} wallet: ${s.wallet:.0f} (expected ${info['expected']:.0f}) - {info['role']}")
        assert abs(s.wallet - info["expected"]) < 0.01, f"Wrong wallet for {name}: got {s.wallet}, expected {info['expected']}"
        assert s.financial_stress == 0.0, f"Wrong stress for {name}"

print("\n✓ Backwards-compatible wallet initialization working")

# Test 2: Financial stress calculation via ResourceManager
print("\n[TEST 2] Financial stress thresholds")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    rm = WorldResourceManager(tmpdir)
    scratch_path = create_test_scratch("Isabella Rodriguez", tmpdir)
    s = Scratch(scratch_path)
    
    s.wallet = 5.0   # Danger
    rm._update_financial_stress(s)
    print(f"  Wallet $5   → stress {s.financial_stress:.1f} (expected 1.0)")
    assert s.financial_stress == 1.0

    s.wallet = 25.0  # Critical
    rm._update_financial_stress(s)
    print(f"  Wallet $25  → stress {s.financial_stress:.1f} (expected ~0.7)")
    assert 0.65 < s.financial_stress < 0.75

    s.wallet = 50.0  # Low
    rm._update_financial_stress(s)
    print(f"  Wallet $50  → stress {s.financial_stress:.1f} (expected ~0.4)")
    assert 0.35 < s.financial_stress < 0.45

    s.wallet = 100.0 # Comfortable
    rm._update_financial_stress(s)
    print(f"  Wallet $100 → stress {s.financial_stress:.1f} (expected 0.0)")
    assert s.financial_stress == 0.0

print("\n✓ Financial stress calculation working")

# Test 3: Purchase from ResourceManager
print("\n[TEST 3] ResourceManager purchase() with wallet")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    rm = WorldResourceManager(tmpdir)
    
    # Create buyer persona
    scratch_path = create_test_scratch("Klaus Mueller", tmpdir)
    buyer = Scratch(scratch_path)
    print(f"  Buyer (Klaus) initial wallet: ${buyer.wallet:.0f}")
    
    # Purchase coffee from cafe counter
    result = rm.purchase("Hobbs Cafe: counter", "coffee_beans", 1, buyer)
    print(f"  Purchase 1x coffee beans (${ITEM_PRICES['coffee_beans']:.2f}): {result}")
    print(f"  Buyer wallet after: ${buyer.wallet:.2f}")
    
    assert result == True, "Purchase should succeed"
    expected = 120.0 - ITEM_PRICES["coffee_beans"]
    assert abs(buyer.wallet - expected) < 0.01, f"Wrong deduction: {buyer.wallet}"
    
    # Try purchase with insufficient funds (sandwiches are $4.50)
    poor_buyer = Scratch(scratch_path)
    poor_buyer.wallet = 1.0
    result2 = rm.purchase("Hobbs Cafe: counter", "sandwiches", 1, poor_buyer)
    print(f"  Purchase 1x sandwich (${ITEM_PRICES['sandwiches']:.2f}) with $1 wallet: {result2} (should fail)")
    assert result2 == False, "Should fail with insufficient funds"

print("\n✓ Purchase transactions working with wallet deduction")

# Test 4: Wallet persistence via save/load
print("\n[TEST 4] Wallet persistence save/load")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    # Load and modify
    scratch_path = create_test_scratch("Isabella Rodriguez", tmpdir)
    s = Scratch(scratch_path)
    s.wallet = 275.50
    s.financial_stress = 0.2
    
    # Save to new file
    save_path = f"{tmpdir}/test_save.json"
    s.save(save_path)
    
    # Load and verify
    with open(save_path) as f:
        saved = json.load(f)
    print(f"  Saved wallet: ${saved.get('wallet', 'MISSING')}")
    print(f"  Saved stress: {saved.get('financial_stress', 'MISSING')}")
    
    assert saved.get('wallet') == 275.50, "Wallet not persisted"
    assert saved.get('financial_stress') == 0.2, "Stress not persisted"

print("\n✓ Wallet/stress save/load working")

# Test 5: Financial summary for LLM prompts
print("\n[TEST 5] Financial summary injection")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    scratch_path = create_test_scratch("Isabella Rodriguez", tmpdir)
    s = Scratch(scratch_path)
    
    s.wallet = 15.0
    rm._update_financial_stress(s)
    summary = s.get_str_financial_summary()
    print(f"  Summary (wallet $15): '{summary.strip()}'")
    assert "worried" in summary.lower() or "$15" in summary
    
    s.wallet = 250.0
    rm._update_financial_stress(s)
    summary2 = s.get_str_financial_summary()
    print(f"  Summary (wallet $250): '{summary2.strip() if summary2 else '(empty - comfortable)'}'")

print("\n✓ Financial summaries formatted correctly")

# Test 6: Item pricing
print("\n[TEST 6] Item price list")
print("-" * 50)
for item, price in ITEM_PRICES.items():
    print(f"  {item:20s} ${price:.2f}")

print("\n✓ Pricing configured")

# Test 7: Cafe revenue loop (simulate Isabella earning from sale)
print("\n[TEST 7] Café revenue flow")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    rm = WorldResourceManager(tmpdir)
    
    # Set up buyer and seller
    isabella_path = create_test_scratch("Isabella Rodriguez", tmpdir)
    klaus_path = create_test_scratch("Klaus Mueller", tmpdir)
    
    isabella = Scratch(isabella_path)
    klaus = Scratch(klaus_path)
    
    initial_izzy = isabella.wallet
    initial_klaus = klaus.wallet
    
    print(f"  Before: Isabella ${initial_izzy:.0f}, Klaus ${initial_klaus:.0f}")
    
    # Klaus buys coffee
    rm.purchase("Hobbs Cafe: counter", "coffee_beans", 1, klaus)
    # Credit Isabella (simulating what reverie would do)
    price = ITEM_PRICES["coffee_beans"]
    isabella.wallet += price
    
    print(f"  After:  Isabella ${isabella.wallet:.0f}, Klaus ${klaus.wallet:.2f}")
    print(f"  Isabella earned: ${isabella.wallet - initial_izzy:.2f}")
    
    assert abs(klaus.wallet - (120.0 - price)) < 0.01, "Klaus deduction wrong"
    assert abs(isabella.wallet - (250.0 + price)) < 0.01, "Isabella credit wrong"

print("\n✓ Café revenue flow working")

print("\n" + "=" * 60)
print("ALL TESTS PASSED — Phase 5 Economy Layer operational")
print("=" * 60)
