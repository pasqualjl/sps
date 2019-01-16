import pandas as pd
from scipy.stats import ttest_ind

rr = pd.read_csv("new.csv")

f = rr[rr["label"] == 0]
t = rr[rr["label"] == 1]

print("\n")
print("acceleration:", ttest_ind(f['acc'], t['acc']))
print("x-axis:", ttest_ind(f['ax_f'], t['ax_f']))
print("y-axis:", ttest_ind(f['ay_f'], t['ay_f']))
print("z-axis:", ttest_ind(f['az_f'], t['az_f']))
print("costheta:", ttest_ind(f['costheta'], t['costheta']))
print("jerk:", ttest_ind(f['jerk'], t['jerk']))
print("\n")