from gem5.simulator import PieEnvironment
from gem5 import simulator
import atexit

# Initialize the PIE environment with descriptive tagging
env = simulator.make(
    timeout_seconds_gem5=120,
    verbose=True,
    use_logical_cpus=True,
    port=10086,
    workers=30,
    exit_early_on_fail=True,
    api_key="!EWQ2131d"
)

@atexit.register
def cleanup():
    env.teardown()

# Log environment information
env.log()
