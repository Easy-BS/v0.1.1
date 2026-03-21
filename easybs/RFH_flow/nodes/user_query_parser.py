# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 07:09:03 2025

@author: Xiguan Liang @SKKU
"""
#./RFH_flow/nodes/user_query_parser.py
from state_schema import SimulationState

def user_query_parser(state: SimulationState) -> SimulationState:
    # Just normalize the input, like your Multi_flow node
    state["user_input"] = (state.get("user_input") or "").strip()
    if not state["user_input"]:
        return {"errors": ["No user input received."]}
    return state
