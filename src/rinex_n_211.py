import re  # regular expression
import matplotlib.pyplot as plt # type: ignore # matplot for plot data
import os
from datetime import datetime  # date time to convert epoch-time
from collections import defaultdict # use for store data


rinex_file = r"../data/brdc1810.o9n"  # Use uploaded file path

output_dir = "epoch_plots"
os.makedirs(output_dir, exist_ok=True)

header_data = {}  # dictionary to store header data
types_of_obs = [] # list of observation types
obs_data = defaultdict(list)  # This will store data with the epoch time as the key and a list of tuples as values

def convert_epoch_time(epoch_time):
    parts = epoch_time.split()
    # Split time from string
    year, month, day, hour, minute = map(int, parts[:5])
    second = float(parts[5]) # Because seconds value are float so we need to convert it separately
    # Convert days, months, years, hours, minutes, seconds to new format
    formatted_time = datetime(year + 2000, month, day, hour, minute, int(second)).strftime("%H:%M:%S %d/%m/%Y")
    # Convert again for datetime type 
    converted_epoch_time = datetime.strptime(formatted_time, "%H:%M:%S %d/%m/%Y")
    return converted_epoch_time 



