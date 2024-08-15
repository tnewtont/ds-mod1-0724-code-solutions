# -*- coding: utf-8 -*-
"""visualization-project-laptop-defects

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GEPIN4EhOlpCtMp7WsTiz1qMykNTgruk

# **#1 Graph the number of defects over time.**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

dap = pd.read_excel('/content/Data_Analysis_Project.xlsx', 'Data') # Reading from Data sheet
dap

# Convert Month ID to datetime
dap['Month ID(YYYYMM)'] = pd.to_datetime(dap['Month ID(YYYYMM)'], format = '%Y%m')

# Dividing mean rate and limits by 1,000,000
dap['Mean Rate'] = dap['Mean Rate'] / 1000000
dap['2 Sigma limit'] = dap['2 Sigma limit'] / 1000000
dap['3 Sigma limit'] = dap['3 Sigma limit'] / 1000000

# defect rate = defects / opportunities
dap['Defect Rate'] = dap['Defects'] / dap['Opportunities']

dap

plt.plot('Month ID(YYYYMM)', 'Defects', data = dap) # Using matplotlib

"""# **#2 Graph the number of opportunities over time.**"""

plt.plot('Month ID(YYYYMM)', 'Opportunities', data = dap) # Using matplotlib

"""# **#3 Graph defect rate with mean rate and limits over time.**"""

# defect rate = defects over time
plt.plot('Month ID(YYYYMM)', 'Defect Rate', data = dap, color = 'tab:blue') # Defect Rate
plt.plot('Month ID(YYYYMM)', 'Mean Rate', data = dap, color = 'tab:orange') # Mean rate
plt.plot('Month ID(YYYYMM)', '2 Sigma limit', data = dap, color = 'tab:red') # 2 standard deviations
plt.plot('Month ID(YYYYMM)', '3 Sigma limit', data = dap, color = 'tab:pink') # 3 standard deviations

#MonthID is an int type initially, but it's interpretting as an epoch time, pass in format

"""# **#4 Show all the Information from Task-1, Task-2 and Task-3 in a figure that can be presented to business, being mindful of formatting & clarity.**"""

# Plot 3 is rates, but # Plots 1 and 2 are hard numbers, so add an extra axis

# Hard numbers as y values

# 2-entry tuple, how many plots you want in the figure

# (2,2) returns an array of axes

fig, ax1 = plt.subplots()


ax1.set_xlabel('time (YYYYMM)')
ax1.set_ylabel('raw values')
ax1.plot('Month ID(YYYYMM)', 'Defects', data = dap, color = 'navy') # Using matplotlib (0 < y < 250)
ax1.plot('Month ID(YYYYMM)', 'Opportunities', data = dap, color = 'darkgreen') # Using matplotlib (0 < y < 3500)

# 1st annotation
rapid_increase = ax1.text(dt.datetime(2016, 11, 1), 1500, "rapid increase",
            ha="center", va="center", rotation=70, size=10,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="lightblue", ec="steelblue", lw=2))

ax1.tick_params(axis = 'y')

# Defects, Means, 2sigma, 3 sigma

ax2 = ax1.twinx()
ax2.plot('Month ID(YYYYMM)', 'Defect Rate', data = dap, color = 'saddlebrown') # Defect Rate

# 2nd annotation
peak = ax2.annotate('max peak, then rapid decline', xy = (dt.datetime(2016, 8, 1), 0.722222), xytext = (dt.datetime(2016, 10, 1), 0.722222), arrowprops=dict(facecolor='black', shrink=0.05))

ax2.plot('Month ID(YYYYMM)', 'Mean Rate', data = dap, color = 'dodgerblue') # Mean rate
ax2.plot('Month ID(YYYYMM)', '2 Sigma limit', data = dap, color = 'deeppink') # 2 standard deviations
ax2.plot('Month ID(YYYYMM)', '3 Sigma limit', data = dap, color ='maroon') # 3 standard deviations

# 3rd annotation
steady = ax2.text(dt.datetime(2017, 9, 1), 0.15, "defect rate ≈ mean rate",
            ha="center", va="center", rotation=0, size=18,
            bbox=dict(boxstyle="darrow,pad=0.005",
                      fc="lawngreen", ec="olivedrab", lw=2))
ax2.set_ylabel('rates')
ax2.tick_params(axis = 'y')

fig.tight_layout() # Prevent y axis from clipping
plt.show()

"""# **#5 Annotate at least three observations in the data provided.**"""

# Max peak at (2016-08-01	, 0.722222) then rapid decline therafter, for defect rate
# Between 2016-07 and 2017-01, there is a rapid increase in the number of opportunities overtime
# At around 2016-12 and onward, remains roughly steady with defect rate ≈ mean rate