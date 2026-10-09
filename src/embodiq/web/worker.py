"""A separate main thread owns pipeline signals; never run it in HTTP threads."""
import sys
from embodiq.pipeline import execute_pipeline

if __name__ == '__main__':
    # The parent cleared any old stop marker under the dispatch lock. Keeping
    # resume=False preserves a new stop request received during process startup.
    sys.exit(execute_pipeline(sys.argv[1], gpu=int(sys.argv[2]), wait_seconds=int(sys.argv[3])))
