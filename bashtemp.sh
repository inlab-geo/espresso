#!/bin/bash
# This script currently only works if _version.py contains only ONE LINE OF CODE.


var1=$(grep -oP '(?<=0.0.)[0-99]+' contrib/_version.py)

out_ver=$(($var1 + 1))

str="__version__=\"0.0."

out="$str$out_ver\""

sed -i '1s/.*/'$out'/' contrib/_version.py
