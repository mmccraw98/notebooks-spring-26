#!/bin/bash

python initialize.py --mu "0.1" --alpha "1.0"
python initialize.py --mu "0.1" --alpha "1.2"
python initialize.py --mu "0.1" --alpha "1.5"
python initialize.py --mu "0.1" --alpha "2.0"

python initialize.py --mu "0.5" --alpha "1.0"
python initialize.py --mu "0.5" --alpha "1.2"
python initialize.py --mu "0.5" --alpha "1.5"
python initialize.py --mu "0.5" --alpha "2.0"

python initialize.py --mu "1.0" --alpha "1.0"
python initialize.py --mu "1.0" --alpha "1.2"
python initialize.py --mu "1.0" --alpha "1.5"
python initialize.py --mu "1.0" --alpha "2.0"

python initialize.py --mu "0.01" --alpha "1.0"
python initialize.py --mu "0.01" --alpha "1.2"
python initialize.py --mu "0.01" --alpha "1.5"
python initialize.py --mu "0.01" --alpha "2.0"