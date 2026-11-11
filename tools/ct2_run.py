"""Run the tested CTranslate2 backend without changing global packages."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT/'runtime/ct2']:
    if not path.exists():raise SystemExit('Missing local runtime: '+str(path))
    sys.path.insert(0,str(path))
sys.path.insert(0,str(ROOT))
if len(sys.argv)<2 or sys.argv[1] not in {'convert','evaluate','translate'}:
    raise SystemExit('Usage: python tools/ct2_run.py convert|evaluate|translate [options]')
mode=sys.argv.pop(1)
runpy.run_module({'convert':'experiments.convert_ct2','evaluate':'experiments.evaluate_ct2',
                 'translate':'translator.ct2_cli'}[mode],run_name='__main__')
