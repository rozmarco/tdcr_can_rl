import pytest
import numpy as np
from src.buffers.buffer import ReplayBuffer

@pytest.fixture
def buffer():
    """Fixture to provide a fresh buffer for every test."""
    return ReplayBuffer(max_size=200)

def test_add_and_sample_single(buffer):
    # Setup data
    state = np.array([1, 2, 3, 4])
    action = np.array([0])
    reward = 1.0
    next_state = np.array([1, 2, 3, 5])
    done = False

    # Add to buffer
    buffer.add(state, action, reward, next_state, done)
    
    # Sample
    s, a, r, ns, d = buffer.sample(batch_size=1, horizon=1)

    # Check content match
    assert np.array_equal(s[0][0], state)
    assert r[0][0] == 1.0
    assert d[0][0] == False

def test_without_replacement_exhaustion(buffer):
    """Verifies that the sampler stops when indices are exhausted."""
    # Add 2 items
    for i in range(2):
        buffer.add(np.zeros(2), 0, 0.0, np.zeros(2), False)

    # First sample: get 2 items
    s, _, _, _, _ = buffer.sample(batch_size=2)
    assert len(s) == 2

    # Second sample: should be empty/None because we haven't reset 
    # and we are sampling without replacement
    s2, _, _, _, _ = buffer.sample(batch_size=1)
    assert len(s2) == 0

def test_horizon_logic(buffer):
    """Tests that horizon correctly packages sequences."""
    # Add 3 sequential items
    for i in range(3):
        buffer.add(np.array([i]), 0, float(i), np.array([i+1]), False)
    
    # Sample with horizon 3
    # Note: This depends on the sampler picking index 0
    # To be safe for testing, we can manually set the sampler indices
    buffer.sampler.indices = np.array([0]) 
    
    s, a, r, ns, d = buffer.sample(batch_size=1, horizon=3)
    
    # Sequence length should be 3
    assert len(s[0]) == 3
    assert s[0][0] == np.array([0])
    assert s[0][2] == np.array([2])

def test_done_termination(buffer):
    """
    Tests that the sequence truncates at 'done' and 'done' flag correctly 
    stops a sequence, even if there is data from a new episode immediately 
    following it.
    """
    # Add: [Step 0, not done], [Step 1, DONE], [Step 2, not done]
    buffer.add(np.array([0]), 0, 0.0, np.array([1]), False)
    buffer.add(np.array([1]), 0, 0.0, np.array([2]), True)
    buffer.add(np.array([2]), 0, 0.0, np.array([3]), False)

    # Force sampler to start at index 0
    buffer.sampler.indices = np.array([0])
    s, _, _, _, d = buffer.sample(batch_size=1, horizon=3)

    # Should truncate to length 2 because index 1 was 'done'
    assert len(s[0]) == 2
    assert d[0][-1] == True

    # The first state of Episode 2 (20) should NOT be in this batch
    for seq in s[0]:
        assert seq[0] < 20

def test_circular_wrap_around(buffer):
    """
    Tests that the buffer overwrites the oldest data when max_size is exceeded.
    """
    buffer.max_size = 5
    
    # Fill buffer to capacity (size 5)
    for i in range(5):
        buffer.add(np.array([i]), 0, 0.0, np.array([i+1]), False)
    
    assert buffer.size == 5
    
    # Add 2 more items (should overwrite indices 0 and 1)
    # New data: [100], [101]
    buffer.add(np.array([100]), 0, 0.0, np.array([101]), False)
    buffer.add(np.array([101]), 0, 0.0, np.array([102]), False)
    
    # Since it's circular, the 'state' array at index 0 should now be 100
    assert buffer.state[0][0] == 100
    assert buffer.state[1][0] == 101
    assert buffer.state[2][0] == 2  # Old data still there

def test_reload_and_resample(buffer):
    """
    Tests: 
    1. Load 100, sample 100 (exhaustion).
    2. Load 100 more, sample 100 (reload).
    """
    # 1. Load first 100
    for i in range(100):
        buffer.add(np.array([i]), 0, 0.0, np.array([i]), False)
    
    # Sample all 100
    s1, _, _, _, _ = buffer.sample(batch_size=100, horizon=1)
    assert len(s1) == 100
    
    # Verify exhaustion
    s_empty, _, _, _, _ = buffer.sample(batch_size=1, horizon=1)
    assert len(s_empty) == 0

    # Reset when exhausted, then add more data
    buffer.sampler.reset()

    # 2. Load next 100 (Buffer size is now 200)
    for i in range(100, 200):
        buffer.add(np.array([i]), 0, 0.0, np.array([i]), False)

    # 3. Sample again
    s2, _, _, _, _ = buffer.sample(batch_size=100, horizon=1)
    
    assert len(s2) == 100
    flattened_s2 = [val[0][0] for val in s2]
    assert all(val >= 0 for val in flattened_s2)