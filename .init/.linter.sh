#!/bin/bash
cd /home/kavia/workspace/code-generation/multiplayer-tic-tac-toe-90ae4ea1/tic_tac_toe_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

