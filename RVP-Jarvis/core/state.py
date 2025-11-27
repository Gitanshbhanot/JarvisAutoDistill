import json
import fcntl
from pathlib import Path

class AppState:
    def __init__(self):
        self.state_file = Path("data/app_state.json")
        self.state_file.parent.mkdir(exist_ok=True)
        self._initialize_state()

    def _initialize_state(self):
        """Initialize state file if it doesn't exist."""
        default_state = {
            "current_timestamp": None,
            "annotation_status": "pending",
            "workflow_mode": "annotate",
            "annotation_progress": [],
            "annotation_total": 0,
            "annotation_current": 0,
            "cancelled": False
        }
        if not self.state_file.exists():
            with open(self.state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(default_state, f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _read_state(self):
        """Read state from JSON file with locking."""
        try:
            with open(self.state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                state = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return state
        except (json.JSONDecodeError, FileNotFoundError):
            self._initialize_state()
            return self._read_state()

    def _write_state(self, state):
        """Write state to JSON file with locking."""
        with open(self.state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(state, f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def __getattr__(self, name):
        state = self._read_state()
        return state.get(name, None)

    def __setattr__(self, name, value):
        if name == "state_file":
            super().__setattr__(name, value)
        else:
            state = self._read_state()
            state[name] = value
            self._write_state(state)

# Initialize global state
app_state = AppState()