import os
import sys
import traceback
import locale
sys.stdout.reconfigure(encoding='utf-8')

try:
    repo_path=os.path.abspath('.')
    if repo_path not in sys.path:
        sys.path.append(repo_path)
    from route_engine import DRLRouteEngine
    engine = DRLRouteEngine(os.path.abspath('my_checkpoints_distance'))
    print(engine.compute_optimal_route('50317015#0', '24884043#18'))
except Exception as e:
    traceback.print_exc()
