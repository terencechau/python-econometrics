import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyfixest as pf
import re

from event_study_functions import *

event_study_simultaneous = run_dynamic_did("output/panel_data_simultaneous.csv")
event_study_staggered = run_dynamic_did("output/panel_data_staggered.csv")

pf.etable([event_study_staggered, event_study_simultaneous])

plot_event_study(event_study_simultaneous)
plot_event_study(event_study_simultaneous, ribbons=True)
    
