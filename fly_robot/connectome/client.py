"""neuprint client setup for the MaleCNS connectome (v1.0).

Dataset choice: MaleCNS, not MANC. MaleCNS is a distinct, newer specimen
(brain + VNC imaged as one continuous volume, incl. the neck connective)
released 2026; MANC (2023) is nerve-cord-only and is what the Pugliese et
al. CPG model was built on. We use MaleCNS going forward for our own
neuron identification and circuit extraction; MANC is only relevant if we
later re-run Pugliese's exact pipeline for validation.
"""

import os

from dotenv import load_dotenv
from neuprint import Client

NEUPRINT_SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

_client = None


def get_client() -> Client:
    """Return a cached, authenticated neuprint Client for MaleCNS v1.0."""
    global _client
    if _client is None:
        load_dotenv()
        token = os.environ.get("NEUPRINT_TOKEN")
        if not token:
            raise RuntimeError(
                "NEUPRINT_TOKEN not set. Add it to a .env file in the repo root "
                "(see README) — get a token from https://neuprint.janelia.org "
                "under your account profile."
            )
        _client = Client(NEUPRINT_SERVER, dataset=DATASET, token=token)
    return _client
