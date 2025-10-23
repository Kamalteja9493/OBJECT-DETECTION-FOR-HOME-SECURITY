#!/bin/bash

echo " Starting Real-Time YOLOv5 Detection Project"

# Optional: Activate virtual environment if used
# source ~/your-env/bin/activate

# Run webcam-based detection
python3 webcam.py

echo " Webcam detection finished."

# Run notebook for comparison (optional, if command-line execution is expected)
# jupyter nbconvert --to notebook --execute "notebookce0b99c74d (1).ipynb" --output comparison_output.ipynb

echo " Model comparison notebook is available as: notebookce0b99c74d (1).py"

