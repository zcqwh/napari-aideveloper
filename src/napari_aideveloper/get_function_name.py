#!/usr/bin/env python3
"""
Created on Fri Mar 25 13:54:23 2022

@author: nana
"""

from inspect import getmembers, isfunction

import aid_backbone
import aid_bin
import aid_dl
import aid_img
import aid_start

functions_aid_bin = [o[0] for o in getmembers(aid_bin) if isfunction(o[1])]
functions_aid_dl = [o[0] for o in getmembers(aid_dl) if isfunction(o[1])]
functions_aid_start = [o[0] for o in getmembers(aid_start) if isfunction(o[1])]
functions_aid_img = [o[0] for o in getmembers(aid_img) if isfunction(o[1])]
functions_aid_backbone = [o[0] for o in getmembers(aid_backbone) if isfunction(o[1])]



functions_name = functions_aid_backbone
function_list = ""
for i in functions_name:
    function_list += i + ", "

print(function_list)
